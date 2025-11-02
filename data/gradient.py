import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
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
    def __init__(self, dataset_path, split="train", batch_size=16, name="", shuffle=True, repeat=True):
        self.dataset = load_dataset(dataset_path, split=split)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat

        self.buckets = BUCKET_SIZES
        self.bucket_ratios = [w / h for w, h in self.buckets]
        self.base_transform = transforms.Compose([transforms.ToTensor()])

        self.is_streaming = getattr(self.dataset, "is_streaming", False)

    def _make_iter(self):
        """Return a fresh iterator over samples (shuffled if needed)."""
        if self.is_streaming:
            if self.shuffle:
                return iter(self.dataset.shuffle(buffer_size=1000, seed=random.randint(0, 1e6)))
            else:
                return iter(self.dataset)
        else:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            return (self.dataset[i] for i in indices)

    def __iter__(self):
        bucket_batches = [[] for _ in self.buckets]

        while True:
            dataset_iter = self._make_iter()

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

            if not self.repeat:
                break
    def init_dataloader(self, **kwargs):
        return DataLoader(
            self,
            batch_size=None,  # Batching handled in __iter__
            pin_memory=True,
            num_workers=8,
            **kwargs,
        )