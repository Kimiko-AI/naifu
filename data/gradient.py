import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

class GradientDataset(Dataset):
    def __init__(
        self,
        width,
        height,
        num_batches=100,
        name="",
        start_color=(0., 0., 0.),
        end_color=(1., 1., 0.),
        batch_size = 16
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
        return DataLoader(self, batch_size = self.batch_size, shuffle = True, pin_memory = True, )

BUCKET_SIZES = [
    (256, 256),
    (224, 288),
    (288, 224),
    (320, 192),
    (192, 320),
]



def nearest_bucket(width, height, buckets=BUCKET_SIZES):
    """Return the (w, h) bucket closest in aspect ratio and area."""
    aspect = width / height
    return min(
        buckets,
        key=lambda b: abs((b[0] / b[1]) - aspect) + 0.1 * abs((b[0] * b[1]) - (width * height))
    )


class TagImageDataset(Dataset):
    def __init__(self, dataset_path, split="train", image_size=512, batch_size=16, name=""):
        self.dataset = load_dataset(dataset_path, split=split)
        self.batch_size = batch_size

        # Gather image sizes (width, height)
        self.sizes = []
        for sample in self.dataset:
            img = sample["webp"]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            self.sizes.append(img.size)
            img.close()

        self.base_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        json_data = sample["json"]
        rating = json_data.get("rating", "")
        character_tags = json_data.get("character_tags", [])
        general_tags = json_data.get("general_tags", [])
        all_tags = [rating] + character_tags + general_tags
        tag_str = " ".join(map(str, all_tags))[:512]

        image = sample["webp"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        return image, tag_str, self.sizes[idx]

    # -------------------------------
    # Batch sampler and dataloader
    # -------------------------------
    def init_dataloader(self, **kwargs):
        sampler = BucketBatchSampler(self, batch_size=self.batch_size)
        return DataLoader(
            self,
            batch_sampler=sampler,
            collate_fn=self.bucket_collate_fn,
            pin_memory=True,
            **kwargs,
        )

    @staticmethod
    def bucket_collate_fn(batch):
        """Resize batch to its bucket size and stack tensors."""
        images, tags, sizes = zip(*batch)
        # Pick the first sample's bucket (all in same bucket)
        target_bucket = nearest_bucket(*sizes[0])
        resize = transforms.Resize(target_bucket[::-1])  # (h, w)
        tensor_images = torch.stack([transforms.ToTensor()(resize(img)) for img in images])
        return {"pixels": tensor_images, "prompts": list(tags)}


class BucketBatchSampler(Sampler):
    """Groups dataset indices by similar-size bucket."""
    def __init__(self, dataset, batch_size=16):
        self.batch_size = batch_size
        self.buckets = {b: [] for b in BUCKET_SIZES}

        # Assign indices to buckets
        for idx, (w, h) in enumerate(dataset.sizes):
            bucket = nearest_bucket(w, h)
            self.buckets[bucket].append(idx)

        # Prepare batches grouped by bucket
        self.batches = []
        for b, idxs in self.buckets.items():
            for i in range(0, len(idxs), batch_size):
                self.batches.append(idxs[i:i + batch_size])

    def __iter__(self):
        # Optional shuffle for randomness
        torch.random.manual_seed(torch.randint(0, 2**31, (1,)).item())
        shuffled_batches = torch.randperm(len(self.batches)).tolist()
        for b_idx in shuffled_batches:
            yield self.batches[b_idx]

    def __len__(self):
        return len(self.batches)