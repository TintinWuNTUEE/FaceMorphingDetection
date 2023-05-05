import os
import torch
from torch import Generator as G
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as T
from PIL import Image
from common.configs import get_cfg_defaults
from data import FaceDataset, FaceSubset


def kfold(cfg, train_data, i):
    k_fold = cfg.train.k_folds
    total_size = len(train_data)
    val_indices = []
    train_indices = []
    for t in range(total_size):
        if t % k_fold == i:
            val_indices.append(t)
        else:
            train_indices.append(t)
    print(k_fold, i, len(val_indices), len(train_indices))
    val_set = torch.utils.data.dataset.Subset(train_data, val_indices)
    train_set = torch.utils.data.dataset.Subset(train_data, train_indices)

    return val_set, train_set


def get_loader(image_dir, val_num, batch_size=16, num_workers=8):
    """Build and return a data loader."""
    train_transform, test_transform = get_transform()
    dataset = FaceDataset(image_dir)
    P = 0.8
    lengths = [int(len(dataset) * P), len(dataset) - int(len(dataset) * P)]

    train_data, test_data = random_split(
        dataset, lengths, generator=G().manual_seed(666)
    )

    train_data, test_data = FaceSubset(train_data, train_transform), FaceSubset(
        test_data, test_transform
    )
    val_set, train_set = kfold(cfg, train_data, val_num)

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, val_loader


def get_transform():
    train_transform = []
    train_transform.append(
        T.Resize((260, 260), interpolation=T.InterpolationMode.BICUBIC)
    )
    train_transform.append(T.CenterCrop((256, 256)))
    train_transform.append(T.RandAugment(interpolation=T.InterpolationMode.BICUBIC))
    train_transform.append(T.ToTensor())
    train_transform.append(
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    train_transform = T.Compose(train_transform)

    test_transform = []
    test_transform.append(
        T.Resize((260, 260), interpolation=T.InterpolationMode.BICUBIC)
    )
    test_transform.append(T.CenterCrop((256, 256)))
    test_transform.append(T.ToTensor())
    test_transform.append(
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return train_transform, test_transform


class FaceDataset(Dataset):
    """
    Face parent dataset
    """

    def __init__(self, root):
        self.root = root
        self.filenames = []
        self.filenames = self.read_image(self.root)

    def read_image(self, path):
        filepaths = []
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        for root, _, files in os.walk(path):
            for filename in files:
                if any(filename.endswith(ext) for ext in image_extensions):
                    filepath = os.path.join(root, filename)
                    filepaths.append(filepath)
        return filepaths

    def __len__(self):
        return len(self.filenames)

    def get_label(self, img_name):
        label = int(img_name.split("_")[-1].split(".")[0])
        return label

    def __getitem__(self, index):
        file = self.filenames[index]
        img = Image.open(file)
        label = self.get_label(file)
        return img, label


class FaceSubset(FaceDataset):
    """
    Subset for different transformation between train test split
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    image_dir = "./data/img_align_celeba"
    val_num = 1
    train_loader, test_loader, val_loader = get_loader(
        image_dir, val_num, num_workers=0
    )
    dataiter = iter(train_loader)
    train_data = next(dataiter)
    train_images, train_label = train_data

    dataiter = iter(val_loader)
    val_data = next(dataiter)
    val_images, val_label = val_data

    print("Image tensor in each batch:", train_images.shape, train_images.dtype)
    print("Label tensor in each batch:", train_label.shape, train_label.dtype)
    print("Image tensor in each batch:", val_images.shape, val_images.dtype)
    print("Label tensor in each batch:", val_label.shape, val_label.dtype)
