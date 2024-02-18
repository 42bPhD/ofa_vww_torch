import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

#Get mean ans std of dataset
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

# 학습 및 검증 데이터셋에 대한 사용자 정의 서브셋
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)

def get_dataloader(
    dataset_dir: str = "vw_coco2014_96",
    batch_size: int = 16,
    image_size: int = 96,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader for training data.

    Parameters
    ----------
    cifar_10_dir: str
        Path to CIFAR10 data root in torchvision format.
    batch_size: int
        Batch size for dataloader.
    num_workers: int
        Number of subprocesses for data loading.
    shuffle: bool
        Flag for shuffling training data.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader for training data.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

        
    # 원본 데이터셋 로드 (변환 없이)
    dataset = ImageFolder(dataset_dir)
        
    # 데이터셋 분할
    validation_split = 0.1
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_indices, val_indices = random_split(dataset, [train_size, val_size])

        
    # 변환을 적용한 데이터셋 생성
    train_dataset = CustomDataset(dataset, train_indices, transform=train_transform)
    val_dataset = CustomDataset(dataset, val_indices, transform=val_transform)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader
