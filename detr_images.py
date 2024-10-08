import json
import h5py
import argparse
import shutil
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


#TODO 切换使用DINO
def collate_crops(data):
    orig_image, captions, idx = zip(*data)
    orig_image = torch.stack(list(orig_image), dim=0)
    captions = list(captions)
    idx = torch.LongTensor(list(idx))
    return orig_image, captions, idx

class Flickr30kImageCrops(Dataset):
    def __init__(self, ann_file, img_root, transform=None):
        self.transform = transform
        self.data = self.parse(Path(ann_file), Path(img_root))

    @staticmethod
    def parse(ann_file, img_root):
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        data = []
        for i, img_info in enumerate(annotations['images']):
            img_file = img_root / img_info['filename']
            captions = [sent['raw'].strip() for sent in img_info['sentences']]
            data.append({
                "image_id": i,
                "image_file": img_file,
                "captions": captions
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]["image_file"])
        image = image.convert("RGB")
        if self.transform is not None:
            orig_image = self.transform(image)
        else:
            orig_image = transforms.ToTensor()(image)
        captions = self.data[index]["captions"]
        idx = self.data[index]["image_id"]
        return orig_image, captions, idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageFeatureExtractor(LightningModule):
    def __init__(self, detr_model_path, output_path, max_detections):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(detr_model_path, revision="no_timm").to(device)
        self.processor = DetrImageProcessor.from_pretrained(detr_model_path, revision="no_timm", do_rescale=False)
        self.output_path = output_path
        self.max_detections = max_detections

    def test_step(self, batch, batch_idx):
        inputs, _, img_ids = batch
        inputs = self.processor(images=inputs, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            results = self.processor.post_process_object_detection(outputs, threshold=0.85, target_sizes=[(800, 1333)]*len(inputs["pixel_values"]))
        with h5py.File(self.output_path, "a") as f:
            for i, img_id in enumerate(img_ids):
                num_boxes = len(results[i]["boxes"])
                feature_map = outputs.last_hidden_state[i][:num_boxes].cpu().numpy()
                feature_map = feature_map.astype(np.float32)
                n, d = feature_map.shape
                delta = self.max_detections - n
                if delta > 0:
                    p = np.zeros((delta, d), dtype=feature_map.dtype)
                    feature_map = np.concatenate([feature_map, p], axis=0)
                elif delta < 0:
                    feature_map = feature_map[:self.max_detections]
                grp = f.create_group(str(img_id.item()))
                grp.create_dataset("obj_features", data=feature_map, compression="gzip")

def get_data_loader(ann_file, image_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((800, 1333)),
        transforms.ToTensor()
    ])
    dataset = Flickr30kImageCrops(ann_file, image_dir, transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_crops
    )
    return data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract image features using DETR')
    parser.add_argument('--exp_name', type=str, default='detr_image_features')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default='/root/autodl-tmp/IC/datasets/Flickr30k')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--detr_model_path', type=str, default='/root/autodl-tmp/IC/detr-resnet101')
    parser.add_argument('--output_path', type=str, default='/root/autodl-tmp/IC/Before_out/oscar_detr.hdf5')
    parser.add_argument('--max_detections', type=int, default=55)
    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("/root/autodl-tmp/IC/Before_out") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    seed_everything(1, workers=True)

    data_loader = get_data_loader(args.dataset_root / "dataset_flickr30k.json",
                                  args.dataset_root / "flickr30k-images" / "flickr30k-images",
                                  args.batch_size,
                                  args.num_workers)

    img_feature_extractor = ImageFeatureExtractor(args.detr_model_path, args.output_path, args.max_detections)

    trainer = Trainer(
        accelerator='gpu',
        devices=[args.device],
        deterministic=False,
        benchmark=False,
        default_root_dir=args.save_dir
    )
    trainer.test(img_feature_extractor, data_loader)
