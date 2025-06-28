from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
import random
import torch
import torchvision.transforms as T

class DatasetGauss(Dataset):
    def __init__(self, root_dir, noise_level="noisy35", patch_size=128, patches_per_image=10, augment=True):
        self.root_dir = Path(root_dir)
        self.clean_dir = self.root_dir / "target"
        self.noise_dir = self.root_dir / noise_level

        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment

        self.image_pairs = []
        for noisy_path in self.noise_dir.glob("*"):
            clean_path = self.clean_dir / noisy_path.name
            if clean_path.exists():
                self.image_pairs.append((noisy_path, clean_path))

        self.to_tensor = T.ToTensor()

    def __len__(self):
        # Total number of samples = number of images × patches per image
        return len(self.image_pairs) * self.patches_per_image

    def __getitem__(self, idx):
        # Get image index and patch index within the image
        img_idx = idx // self.patches_per_image
        noisy_path, clean_path = self.image_pairs[img_idx]

        # Load images and convert to tensors
        noisy_img = Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        noisy = self.to_tensor(noisy_img)
        clean = self.to_tensor(clean_img)

        _, H, W = noisy.shape
        ps = self.patch_size

        if H < ps or W < ps:
            raise ValueError(f"Image too small to crop: {noisy_path.name}")

        # === RANDOM CROP ===
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        noisy_patch = noisy[:, top:top+ps, left:left+ps]
        clean_patch = clean[:, top:top+ps, left:left+ps]

        # === DATA AUGMENTATION ===
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                noisy_patch = TF.hflip(noisy_patch)
                clean_patch = TF.hflip(clean_patch)

            # Vertical flip
            if random.random() < 0.5:
                noisy_patch = TF.vflip(noisy_patch)
                clean_patch = TF.vflip(clean_patch)

            # Rotation (0, 90, 180, 270)
            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                noisy_patch = TF.rotate(noisy_patch, angle)
                clean_patch = TF.rotate(clean_patch, angle)

        return noisy_patch, clean_patch, noisy_path.name
