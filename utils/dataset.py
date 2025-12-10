import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor

        self.images = sorted(os.listdir(image_dir))
        self.masks  = sorted(os.listdir(mask_dir))

        assert len(self.images) == len(self.masks), "Image/mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert mask to integer labels
        mask = np.array(mask).astype("int64")

        inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")

        # HF returns tensors with extra batch dim, we remove manually
        inputs["pixel_values"] = inputs["pixel_values"].squeeze(0)
        inputs["labels"] = inputs["labels"].squeeze(0)
        inputs['filename'] = self.images[idx]

        return inputs
    