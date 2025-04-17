import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class HeartMRIDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.image_dir = os.path.join(data_dir, f"imagesTr")
        self.label_dir = os.path.join(data_dir, f"labelsTr")
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz")])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith(".nii.gz")])

        # Basic train/val split
        total = len(self.image_files)
        cutoff = int(0.8 * total)
        if split == 'train':
            self.image_files = self.image_files[:cutoff]
            self.label_files = self.label_files[:cutoff]
        else:
            self.image_files = self.image_files[cutoff:]
            self.label_files = self.label_files[cutoff:]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = nib.load(img_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.float32)

        # Normalize image (z-score)
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        if self.transform:
            image, label = self.transform(image, label)

        return torch.tensor(image), torch.tensor(label)

# Example usage
if __name__ == "__main__":
    dataset = HeartMRIDataset("/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart")
    print("Dataset size:", len(dataset))
    img, lbl = dataset[0]
    print("Image shape:", img.shape, "Label shape:", lbl.shape)
