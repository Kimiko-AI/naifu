import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import random

class GradientDataset(Dataset):
    def __init__(
            self,
            width,
            height,
            num_batches=100,
            name="",
            start_color=(0., 0., 0.),
            end_color=(1., 1., 0.),
            batch_size=16
    ):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.num_samples = num_batches
        self.start_color = np.array(start_color, dtype=np.float32)
        self.end_color = np.array(end_color, dtype=np.float32)
        self.caption = f"a gradient of {start_color} to {end_color}"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(self.height):
            for j in range(self.width):
                ratio = (i / (self.height - 1) + j / (self.width - 1)) / 2
                color = self.start_color * (1 - ratio) + self.end_color * ratio
                image[i, j] = color
        image = torch.from_numpy(image).permute(2, 0, 1)
        return {"pixels": image, "prompts": self.caption}

    def init_dataloader(self, **kwargs):
        # kwargs can contain batch_size, shuffle, etc.
        return DataLoader(self, batch_size=self.batch_size, shuffle=True, pin_memory=True, )

BUCKET_SIZES = [
    (256, 256),
    (224, 288),
    (288, 224),
    (320, 192),
    (192, 320),
]

class TagImageIterableDataset(IterableDataset):
    def __init__(self, dataset_path, split="train", batch_size=16, name="", shuffle=True):
        self.dataset = load_dataset(dataset_path, split=split)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.buckets = BUCKET_SIZES
        self.bucket_ratios = [w / h for w, h in self.buckets]
        self.base_transform = transforms.Compose([transforms.ToTensor()])
        self.is_streaming = False

    def assign_bucket(self, width, height):
        ratio = width / height
        return min(range(len(self.bucket_ratios)),
                   key=lambda i: abs(self.bucket_ratios[i] - ratio))

    def resize_to_bucket(self, image, bucket_idx):
        target_w, target_h = self.buckets[bucket_idx]
        return image.resize((target_w, target_h), Image.BICUBIC)

    def preprocess_sample(self, sample):
        json_data = sample["json"]
        rating = json_data.get("rating", "")
        character_tags = json_data.get("character_tags", [])
        general_tags = json_data.get("general_tags", [])
        all_tags = [rating] + character_tags + general_tags
        tag_str = " ".join(map(str, all_tags))[:512]

        image = sample["webp"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        return image, tag_str

    def __iter__(self):
        # Create iterable over samples
        if self.is_streaming:
            dataset_iter = iter(self.dataset.shuffle(buffer_size=1000, seed=random.randint(0, 1e6))) if self.shuffle else iter(self.dataset)
        else:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            dataset_iter = (self.dataset[i] for i in indices)

        # Initialize bucket accumulators
        bucket_batches = [[] for _ in self.buckets]

        for sample in dataset_iter:
            try:
                image, tag_str = self.preprocess_sample(sample)
                w, h = image.size
                bucket_idx = self.assign_bucket(w, h)
                image = self.resize_to_bucket(image, bucket_idx)
                image_tensor = self.base_transform(image)

                bucket_batches[bucket_idx].append({
                    "pixels": image_tensor,
                    "prompts": tag_str
                })

                if len(bucket_batches[bucket_idx]) >= self.batch_size:
                    batch = bucket_batches[bucket_idx]
                    pixels = torch.stack([x["pixels"] for x in batch])
                    prompts = [x["prompts"] for x in batch]
                    yield {"pixels": pixels, "prompts": prompts}
                    bucket_batches[bucket_idx] = []
            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue

    def init_dataloader(self, **kwargs):
        return DataLoader(
            self,
            batch_size=None,
            pin_memory=True,
            num_workers=0,
            **kwargs,
        )
