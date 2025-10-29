import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
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



class TagImageDataset(Dataset):
    def __init__(self, dataset_path, split="train", image_size=512, batch_size=16):
        # Load dataset (either local folder or remote HF dataset)
        self.dataset = load_dataset(dataset_path, split=split)
        self.batch_size = batch_size

        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # Converts PIL to (C,H,W) tensor in [0,1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # --- Extract metadata ---
        json_data = sample["json"]
        rating = json_data.get("rating", "")
        #artist_tags = json_data.get("artist_tags", [])
        character_tags = json_data.get("character_tags", [])
        general_tags = json_data.get("general_tags", [])

        # --- Combine tags ---
        all_tags = [rating]  + character_tags + general_tags
        tag_str = " ".join(map(str, all_tags))[:512]

        # --- Load and preprocess image ---
        image = sample["webp"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        image = self.transform(image)

        return {"pixels": image, "prompts": tag_str}

    def init_dataloader(self, **kwargs):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            **kwargs,
        )
