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
        dataset = wds.DataPipeline(
            wds.SimpleShardList(dataset_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.shuffle(10000),
            wds.decode("pil"),
            wds.to_tuple("webp", "json")
        )
        #if shuffle:
        #    dataset = dataset.shuffle(1000)
        #if repeat:
        #    dataset = dataset.repeat()

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
    import torch
    from torch.utils.data import get_worker_info

    # --- Configuration ---
    DATASET_PATH = "/root/ChatError/Dan_dataset/train/{00001..00069}.tar"
    TOTAL_SAMPLES_TO_CHECK = 128 * 32
    BATCH_SIZE = 16
    NUM_WORKERS = 8

    print(f"\nInitializing TagImageIterableDataset from: {DATASET_PATH}")
    print(f"Running check with {TOTAL_SAMPLES_TO_CHECK} samples, "
          f"batch_size={BATCH_SIZE}, num_workers={NUM_WORKERS}\n")

    dataset = TagImageIterableDataset(
        dataset_path=DATASET_PATH,
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=False,
        repeat=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,   # dataset handles batching
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    seen_prompts = set()
    worker_seen = {}

    total_samples = 0
    for batch_idx, batch in enumerate(dataloader):
        prompts = batch["prompts"]
        total_samples += len(prompts)

        # Detect which worker produced this batch
        worker_info = get_worker_info()
        wid = worker_info.id if worker_info else -1

        for p in prompts:
            worker_seen.setdefault(wid, []).append(p)
            if p in seen_prompts:
                print(f"⚠️ Duplicate prompt found in batch {batch_idx}: {p[:60]}...")
            seen_prompts.add(p)

        print(f"Batch {batch_idx:03d}: {len(prompts)} samples "
              f"(total so far: {total_samples})")

        if total_samples >= TOTAL_SAMPLES_TO_CHECK:
            break

    print("\n--- Summary ---")
    print(f"Total unique prompts: {len(seen_prompts)}")
    print(f"Total processed: {total_samples}")
    print(f"Duplicates found: {total_samples - len(seen_prompts)}")

    for wid, samples in worker_seen.items():
        print(f"Worker {wid} handled {len(samples)} samples")

    print("\n✅ Test complete.")
