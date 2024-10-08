import argparse
from pathlib import Path
import shutil
import h5py

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule, seed_everything
import clip

import sys

sys.path.append('.')
from dataset import Flickr30kCaptions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CaptionDB(LightningModule):
    def __init__(self, save_dir):
        super().__init__()

        self.save_dir = save_dir
        self.model, _ = clip.load(args.model, device=device)

    def test_step(self, batch, batch_idx):
        captions, tokens = batch

        # # 确保tokens是整数类型
        # tokens = tokens.long()
        #
        # # 获取模型的权重数据类型
        # dtype = self.model.token_embedding.weight.dtype

        x = self.model.token_embedding(tokens).type(self.model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        keys = self.model.ln_final(x).type(self.model.dtype)
        values = x[torch.arange(len(x)), tokens.argmax(dim=-1)]
        keys = keys[torch.arange(len(x)), tokens.argmax(dim=-1)] @ self.model.text_projection.type(self.model.dtype)
        keys /= keys.norm(dim=-1, keepdim=True)

        values = values.detach().cpu().numpy()
        keys = keys.detach().cpu().numpy()

        with h5py.File(self.save_dir / "caption_db.hdf5", "a") as f:
            g = f.create_group(str(batch_idx))
            g.create_dataset("keys", data=keys, compression="gzip")
            g.create_dataset("values", data=values, compression="gzip")
            g.create_dataset("captions", data=captions, compression="gzip")


def encode_captions(args):
    dset = Flickr30kCaptions(args.ann_file, clip.tokenize)
    dloader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    cap_db = CaptionDB(args.save_dir)

    trainer = Trainer(
        accelerator='gpu',
        devices=[args.device, ],
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir
    )
    trainer.test(cap_db, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode captions')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='captions_db')
    parser.add_argument('--ann_file', type=str, default='/root/autodl-tmp/IC/datasets/Flickr30k/dataset_flickr30k.json')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument(
        "--model", type=str, default="ViT-L/14",
        choices=[
            "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
            "ViT-B/32", "ViT-B/16", "ViT-L/14"
        ]
    )
    args = parser.parse_args()

    setattr(args, "save_dir", Path("/root/autodl-tmp/IC/Before_out") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)

    seed_everything(1, workers=True)

    encode_captions(args)
