import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict

import random
import webdataset as wds
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
    (512, 512),
    (448, 576),
    (576, 448),
    (640, 384),
    (384, 640),
    (320, 704),
    (704, 320)
]
BUCKET_SIZES = [
    (256, 256),
    (224, 288),
    (288, 224),
    (320, 192),
    (192, 320),
]

class TagImageIterableDataset(IterableDataset):
    def __init__(self, dataset_path, split="train", batch_size=16, name="", shuffle=True, repeat=True):
        """
        Iterable dataset for WebDataset archives of (image, JSON metadata) pairs.

        Args:
            dataset_path: Path or URL to WebDataset shards (e.g. "data/{00000..00010}.tar").
            split: Dataset split name, unused but included for compatibility.
            batch_size: Number of samples per bucket batch.
            name: Optional dataset name.
            shuffle: Whether to shuffle the stream (buffered).
            repeat: Whether to repeat endlessly.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.name = name

        # Set up WebDataset pipeline
        dataset = wds.WebDataset(dataset_path).decode("pil").to_tuple("webp", "json")

        if shuffle:
            dataset = dataset.shuffle(1000)
        if repeat:
            dataset = dataset.repeat()

        self.dataset = dataset
        self.buckets = BUCKET_SIZES
        self.bucket_ratios = [w / h for w, h in self.buckets]

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def assign_bucket(self, width, height):
        """Find which bucket an image belongs to based on aspect ratio."""
        ratio = width / height
        return min(range(len(self.bucket_ratios)),
                   key=lambda i: abs(self.bucket_ratios[i] - ratio))

    def resize_to_bucket(self, image, bucket_idx):
        """Resize image to target bucket resolution."""
        target_w, target_h = self.buckets[bucket_idx]
        return image.resize((target_w, target_h), Image.BICUBIC)

    def preprocess_sample(self, sample):
        """Convert WebDataset sample (image, JSON) into usable tensors and text."""
        image, json_data = sample  # because we used .to_tuple("webp", "json")
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        rating = json_data.get("rating", [])
        character_tags = json_data.get("character_tags", [])
        general_tags = json_data.get("general_tags", [])
        all_tags = rating + character_tags + general_tags
        tag_str = " ".join(map(str, all_tags))[:512]

        return image, tag_str

    def _make_iter(self):
        """Return an iterator over the (possibly shuffled) WebDataset."""
        return iter(self.dataset)

    def __iter__(self):
        """Main iteration: bucketed batching from WebDataset stream."""
        bucket_batches = [[] for _ in self.buckets]
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

                # yield a batch when one bucket fills
                if len(bucket_batches[bucket_idx]) >= self.batch_size:
                    batch = bucket_batches[bucket_idx]
                    pixels = torch.stack([x["pixels"] for x in batch])
                    prompts = [x["prompts"] for x in batch]
                    yield {"pixels": pixels, "prompts": prompts}
                    bucket_batches[bucket_idx] = []
            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue

        # no repeat → stop after one full pass
        if not self.repeat:
            return

    def init_dataloader(self, **kwargs):
        """Return a DataLoader wrapping this iterable dataset."""
        return DataLoader(
            self,
            batch_size=None,  # batching handled internally
            pin_memory=True,
            num_workers=8,
            **kwargs,
        )

if __name__ == "__main__":

    # --- Configuration ---
    # !!! IMPORTANT: Replace this with your dataset path or HF repo ID !!!
    # I'm using a public dataset that matches your 'webp' and 'json' fields.
    DATASET_PATH = "/root/ChatError/Dan_dataset/train/{00001..00069}.tar"

    # We want 32 samples. Setting batch_size=32 will make the
    # first batch we get have (up to) 32 samples from one bucket.
    TEST_BATCH_SIZE = 32

    print(f"Initializing dataset from: {DATASET_PATH}")

    # Instantiate the dataset
    # shuffle=False and repeat=False are good for a simple test.
    dataset = TagImageIterableDataset(
        dataset_path=DATASET_PATH,
        split="train",
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        repeat=False
    )

    print("Fetching the first batch of samples...")

    # We use a DataLoader with num_workers=0 for this simple test
    # to avoid multiprocessing issues in a main script.
    # Your init_dataloader() method is for actual training.
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    try:
        # Get the first batch yielded by the iterator
        first_batch = next(iter(dataloader))

        prompts = first_batch["prompts"]

        print(f"\n--- Displaying {len(prompts)} prompts from the first batch ---")

        for i, prompt in enumerate(prompts):
            print(f"Sample {i + 1}:")
            # Adding indentation for readability
            print(f"    {prompt}\n")

            # This check ensures we only print up to 32,
            # even if the batch was somehow larger.
            if i + 1 >= 32:
                break

        print(f"--- Printed {len(prompts)} prompts. ---")

    except StopIteration:
        print("Dataset was empty or too small to produce a single batch.")
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        print("Please check that your DATASET_PATH is correct.")
        print("If using the default, you may need to install 'datasets' and 'Pillow'.")