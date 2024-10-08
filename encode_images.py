import argparse
from pathlib import Path
import h5py
import shutil

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule, seed_everything
import clip

import sys

sys.path.append('.')
from dataset import Flickr30kImageCrops, collate_crops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImageEncoder(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        model, _ = clip.load(args.model, device=device)
        self.model = model.visual

    def test_step(self, batch, batch_idx):
        x, _, ids = batch
        N = len(x)

        if "RN" in self.args.model:
            x = x.type(self.model.conv1.weight.dtype)
            for conv, bn in [(self.model.conv1, self.model.bn1), (self.model.conv2, self.model.bn2),
                             (self.model.conv3, self.model.bn3)]:
                x = self.model.relu(bn(conv(x)))
            x = self.model.avgpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            grid_features = x.flatten(2).transpose(1, 2)
            pooled_features = self.model.attnpool(x)
        else:
            x = x.type(self.model.conv1.weight.dtype)
            x = self.model.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                              dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + self.model.positional_embedding.to(x.dtype)
            x = self.model.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.model.transformer(x)
            grid_features = x.permute(1, 0, 2)  # LND -> NLD
            pooled_features = self.model.ln_post(grid_features[:, 0, :])
            if self.model.proj is not None:
                pooled_features = pooled_features @ self.model.proj

        pooled_features /= pooled_features.norm(dim=-1, keepdim=True)
        pooled_features = pooled_features.detach().cpu().numpy()
        grid_features = grid_features.detach().cpu().numpy()

        with h5py.File(self.args.save_dir / "vis_ctx.hdf5", "a") as f:
            f.attrs["fdim_pooled"] = pooled_features.shape[-1]
            f.attrs["fdim_grid"] = grid_features.shape[-1]
            for i in range(N):
                g = f.create_group(str(int(ids[i])))
                g.create_dataset("pooled_features", data=pooled_features[i], compression="gzip")
                g.create_dataset("grid_features", data=grid_features[i], compression="gzip")


def encode_images(args):
    _, transform = clip.load(args.model, device=device)
    dset = Flickr30kImageCrops(args.dataset_root / "dataset_flickr30k.json", args.dataset_root / "flickr30k-images" / "flickr30k-images", transform)
    dloader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_crops
    )

    img_encoder = ImageEncoder(args)

    trainer = Trainer(
        accelerator='gpu',
        devices=[args.device, ],
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir
    )
    trainer.test(img_encoder, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode images')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='image_features')
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/IC/datasets/Flickr30k')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument("--model", type=str, default="ViT-L/14",
        choices=[
            "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
            "ViT-B/32", "ViT-B/16", "ViT-L/14"
        ]
    )
    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("/root/autodl-tmp/IC/Before_out") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)

    seed_everything(1, workers=True)

    encode_images(args)
