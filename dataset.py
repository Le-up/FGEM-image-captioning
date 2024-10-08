from PIL import Image
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset


#TODO   使用COCO数据集的时候，COCO只进行图片的处理，再下载Visual Genome数据集进行文本的处理，使用COCO的图片数据和V G的文本数据进行检索匹配
#TODO  使用COCO2017

class Flickr30kCaptions(Dataset):
    def __init__(self, ann_file, tokenizer, max_length=55):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.captions, self.tokens = self.parse_annotations(Path(ann_file))

    def parse_annotations(self, ann_file):
        with open(ann_file, "r") as f:
            data = json.load(f)

        captions = []
        tokens = []

        for img in data['images']:
            for sent in img['sentences']:
                caption = sent['raw']
                token = self.tokenizer(caption[:self.max_length]).squeeze(0)  # 截断文本
                captions.append(caption[:self.max_length])  # 确保保存截断后的文本
                tokens.append(token)

        return captions, tokens

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return self.captions[index], self.tokens[index]

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
                "image_id": i,  # 使用索引作为唯一标识符
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
            orig_image = image  # 添加这一行

        captions = self.data[index]["captions"]
        idx = self.data[index]["image_id"]

        return orig_image, captions, idx
