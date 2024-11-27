from torch.nn import functional as F
import copy
from .beam_search import *
from .attention import *
from .captioning_model import CaptioningModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Models(CaptioningModel):
    def __init__(self, bos_idx, m=0.999, K=5000, encoder=None, decoder=None, view_gen=None):
        super(Models, self).__init__()
        self.bos_idx = bos_idx
        self.m = m
        self.K = K

        self.encoder = encoder
        self.decoder = decoder
        self.d_model = self.decoder.d_model
        self.view_gen = view_gen

        self.n_views = self.view_gen.n_views

        self.cls_mlp = self.build_mlp(3, self.d_model, self.d_model*4, self.d_model)
        self.cls_agg = AggregationAttention(self.d_model, self.d_model//8, 8)

        self.ema_models = ["view_gen", "encoder", "cls_mlp", "cls_agg"]
        for m in self.ema_models:
            ema = copy.deepcopy(getattr(self, m))
            for p in ema.parameters():
                p.requires_grad = False
            setattr(self, f"{m}_ema", ema)

        for i in range(self.n_views):
            self.register_state(f"enc_f_{i}", None)
            self.register_state(f"enc_m_{i}", None)
        self.view_cls = None
        self.view_cls_ema = None

        self.register_buffer("queue", F.normalize(torch.randn(self.K, self.d_model), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def ema_update(self):
        for m in self.ema_models:
            model = getattr(self, m)
            model_ema = getattr(self, f"{m}_ema")
            for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def enqueue(self, keys):
        N, n_views, d = keys.shape
        ptr = int(self.queue_ptr)

        keys = keys.reshape(n_views * N, d)
        batch_size = keys.shape[0]
        # assert self.K % batch_size == 0
        self.queue[ptr:ptr + batch_size, :] = keys.detach()

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @property
    def memory(self):
        return self.queue.clone().detach()

    def get_view_cls(self, obj=None, vis_ctx=None, txt_ctx=None, enc_f=None):
        if enc_f is None:
            with torch.no_grad():
                enc_f = self.view_gen_ema(obj, vis_ctx, txt_ctx)
                enc_f = self.encoder_ema(enc_f)[0]
            model = "_ema"
        else:
            model = ""

        with torch.set_grad_enabled(model != "_ema"):
            view_cls = torch.stack([x[:, :, 0] for x in enc_f], dim=1)
            N, n_views, n_layers, d = view_cls.shape

            view_cls = view_cls.reshape(N * n_views * n_layers, d)
            view_cls = getattr(self, f"cls_mlp{model}")(view_cls)
            # assert not torch.isnan(view_cls).any(), "NaN detected in view_cls after MLP"
            view_cls = view_cls.reshape(N, n_views, n_layers, d)

            query = torch.mean(view_cls, dim=2)
            view_cls = getattr(self, f"cls_agg{model}")(query, view_cls)
            # assert not torch.isnan(view_cls).any(), "NaN detected in view_cls after AggregationAttention"

            view_cls = F.normalize(view_cls, dim=-1)
            # assert not torch.isnan(view_cls).any(), "NaN detected in normalized view_cls"

        return view_cls

    def forward_xe(self, obj, vis_ctx, txt_ctx, seq):
        views = self.view_gen(obj, vis_ctx, txt_ctx)

        enc_f, enc_m = self.encoder(views)
        # assert not any(torch.isnan(f).any() for f in enc_f), "NaN detected in encoder features"
        # assert not any(torch.isnan(m).any() for m in enc_m), "NaN detected in encoder memory"

        dec_output = self.decoder(seq, enc_f, enc_m)
        # assert not torch.isnan(dec_output).any(), "NaN detected in decoder output"

        view_cls = self.get_view_cls(enc_f=enc_f)
        view_cls_ema = self.get_view_cls(obj, vis_ctx, txt_ctx)

        return dec_output, view_cls, view_cls_ema

    def forward_rl(self, obj, vis_ctx, txt_ctx, max_len, eos_idx, beam_size, out_size=1, return_probs=False):
        bs = BeamSearch(self, max_len, eos_idx, beam_size)
        bs_ret = bs.apply(obj, vis_ctx, txt_ctx, out_size, return_probs)
        cls_ret = (self.view_cls.clone(), self.view_cls_ema.clone())
        self.view_cls = self.view_cls_ema = None

        return bs_ret, cls_ret

    def forward(self, mode, **kwargs):
        if mode == "xe":
            return self.forward_xe(**kwargs)
        elif mode == "rl":
            return self.forward_rl(**kwargs)
        else:
            raise KeyError

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, obj, vis_ctx, txt_ctx, seq, mode="feedback", **kwargs):
        if mode != "feedback":
            raise NotImplementedError

        it = None
        if t == 0:
            views = self.view_gen(obj, vis_ctx, txt_ctx)
            enc_f, enc_m = self.encoder(views)
            for i in range(self.n_views):
                setattr(self, f"enc_f_{i}", enc_f[i])
                setattr(self, f"enc_m_{i}", enc_m[i])

            self.view_cls = self.get_view_cls(enc_f=enc_f)
            self.view_cls_ema = self.get_view_cls(obj, vis_ctx, txt_ctx)

            it = torch.full((len(obj), 1), self.bos_idx, device=obj.device).long()
        else:
            it = prev_output

        enc_f = [getattr(self, f"enc_f_{i}") for i in range(self.n_views)]
        enc_m = [getattr(self, f"enc_m_{i}") for i in range(self.n_views)]

        return self.decoder(it, enc_f, enc_m)
