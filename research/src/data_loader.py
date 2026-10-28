from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

def image_transform(size, training=False):
    operations = [transforms.Resize((size, size))]
    if training:
        operations += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ColorJitter(.15, .15, .1, .03)]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

class ManifestDataset(Dataset):
    def __init__(self, rows, classes, size=224, training=False):
        self.rows, self.class_to_idx = rows, {v: i for i, v in enumerate(classes)}
        self.transform = image_transform(size, training)
        if any(r["label"] not in self.class_to_idx for r in rows):
            raise ValueError("Manifest contains unknown classes")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(row["path"]) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return tensor, self.class_to_idx[row["label"]]

def make_loader(rows, classes, cfg, training=False):
    ds = ManifestDataset(rows, classes, cfg["image_size"], training)
    # Avoid singleton training batches with BatchNorm, preserving evaluation samples.
    drop_last = training and len(ds) % cfg["batch_size"] == 1
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=training, num_workers=0,
                      drop_last=drop_last, generator=torch.Generator().manual_seed(cfg["seed"]))
