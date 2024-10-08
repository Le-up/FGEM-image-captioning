import torch
import argparse
from pathlib import Path
import h5py
import numpy as np
import math
import faiss
from tqdm import tqdm
import shutil

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule, seed_everything
import clip

from dataset import Flickr30kImageCrops, collate_crops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_float32_matmul_precision('high')
class CaptionRetriever(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_dir = args.save_dir
        self.k = args.k

        self.keys, self.values, self.texts = self.load_caption_db(args.caption_db)
        self.index = self.build_index(idx_file=self.save_dir/"faiss.index")
        self.model, _ = clip.load(args.model, device=device)

    @staticmethod
    def load_caption_db(caption_db):
        keys, values, texts = [], [], []
        with h5py.File(caption_db, "r") as f:
            for i in tqdm(range(len(f))):
                ki = f[f"{i}/keys"][:]
                vi = f[f"{i}/values"][:]
                ti = [str(x, "utf-8") for x in f[f"{i}/captions"][:]]

                keys.append(ki)
                values.append(vi)
                texts.extend(ti)
        keys = np.concatenate(keys)
        values = np.concatenate(values)

        return keys, values, texts

    def build_index(self, idx_file):
        n, d = self.keys.shape
        K = round(8 * math.sqrt(n))
        index = faiss.index_factory(d, f"IVF{K},Flat", faiss.METRIC_INNER_PRODUCT)

        assert not index.is_trained
        index.train(self.keys)
        assert index.is_trained
        index.add(self.keys)
        index.nprobe = max(1, K // 10)

        faiss.write_index(index, str(idx_file))

        return index

    def search(self, images, topk):
        query = self.model.encode_image(images)
        query /= query.norm(dim=-1, keepdim=True)
        D, I = self.index.search(query.detach().cpu().numpy(), topk)

        return D, I

    def test_step(self, batch, batch_idx):
        orig_imgs, _, ids = batch
        N = len(orig_imgs)

        with h5py.File(self.save_dir / "txt_ctx.hdf5", "a") as f:
            D, I = self.search(orig_imgs, topk=8*self.k)

            for i in range(N):
                g = f.create_group(str(int(ids[i])))
                f.attrs["fdim"] = self.values.shape[-1]

                texts = [self.texts[j] for j in I[i]]
                features = self.values[I[i]]
                scores = D[i]
                g1 = g.create_group("whole")
                g1.create_dataset("features", data=features, compression="gzip")
                g1.create_dataset("scores", data=scores, compression="gzip")
                g1.create_dataset("texts", data=texts, compression="gzip")

def retrieve_captions(args):
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
    cap_retr = CaptionRetriever(args)

    trainer = Trainer(
        accelerator='gpu',
        devices=[args.device, ],
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir
    )
    trainer.test(cap_retr, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve captions')
    parser.add_argument('--exp_name', type=str, default='retrieved_captions')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/IC/datasets/Flickr30k')
    parser.add_argument('--caption_db', type=str, default="/root/autodl-tmp/IC/Before_out/captions_db/caption_db.hdf5")
    parser.add_argument('--k', type=int, default=16)
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

    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("/root/autodl-tmp/IC/Before_out") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)

    seed_everything(1, workers=True)

    retrieve_captions(args)
