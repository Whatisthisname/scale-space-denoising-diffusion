import math
import os
import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def pick_device(preferred: str = "auto") -> torch.device:
    """Resolve the fastest available backend, or honour an explicit choice."""
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model, run_name) -> str:
    checkpoint_folder = "checkpoints/" + run_name
    # get all checkpoints and choose the latest:
    checkpoints = os.listdir(checkpoint_folder)
    checkpoints = [c for c in checkpoints if c.endswith(".pth")]
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    latest = checkpoints[-1]

    # load the checkpoint, staging on CPU so a run trained on one backend loads on another:
    data = torch.load(os.path.join(checkpoint_folder, latest), map_location="cpu")
    model.load_state_dict(data)

    return model


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=0):

    def map(x):
        newRange = (-3, 3)
        width = newRange[1] - newRange[0]
        return width * x - width / 2.0

    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            # rescale the images from [0, 1] to [-1, 1] range with a linear transformation
            transforms.Normalize((0.0), (1.0)),
        ]
    )
    # transforms.Lambda(map)])

    train_dataset = MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)

    test_dataset = MNIST(root="./mnist_data", train=False, download=True, transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


class TensorBatches:
    """Iterates minibatches by slicing one resident tensor.

    MNIST is small enough to keep decoded and resized on the accelerator, so the
    per-sample PIL decode that torchvision.MNIST performs every epoch is done once
    here instead. Exposes __len__ so tqdm still shows a progress bar.
    """

    def __init__(self, images, labels, batch_size, shuffle=True, drop_last=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.images)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.images)
        if self.shuffle:
            order = torch.randperm(n, device=self.images.device)
        else:
            order = torch.arange(n, device=self.images.device)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                return
            yield self.images[idx], self.labels[idx]


def load_mnist_split(train: bool, image_size: int) -> tuple:
    """Decode and resize a whole MNIST split once, into [0, 1] floats on the CPU."""
    dataset = MNIST(root="./mnist_data", train=train, download=True)
    images = dataset.data.unsqueeze(1)
    if image_size != images.shape[-1]:
        images = TF.resize(images, [image_size, image_size], antialias=True)
    return images.float().div_(255.0), dataset.targets.clone()


_MNIST_STATS = {}


def mnist_stats(image_size: int) -> tuple:
    """Mean and std of the MNIST train split at a given resolution.

    Resolution matters: downscaling averages pixels and shrinks the spread, so 14x14 MNIST
    has a visibly smaller std than 28x28. Sampling code needs the same numbers as training
    to undo the normalization, so they live here rather than being hardcoded.
    """
    if image_size not in _MNIST_STATS:
        images, _ = load_mnist_split(True, image_size)
        _MNIST_STATS[image_size] = (images.mean().item(), images.std().item())
    return _MNIST_STATS[image_size]


def normalize_images(images: torch.Tensor, image_size: int = None) -> torch.Tensor:
    """Map [0, 1] images to zero mean and unit variance, matching the gaussian the forward process targets."""
    mean, std = mnist_stats(image_size or images.shape[-1])
    return (images - mean) / std


def denormalize_images(images: torch.Tensor, image_size: int = None, clamp: bool = True) -> torch.Tensor:
    """Inverse of normalize_images. Clamping is only appropriate for final output, not intermediates."""
    mean, std = mnist_stats(image_size or images.shape[-1])
    images = images * std + mean
    return images.clamp(0.0, 1.0) if clamp else images


def create_mnist_tensor_loaders(batch_size, image_size=28, device=None, drop_last=False, normalize=True):
    """Drop-in replacement for create_mnist_dataloaders that avoids per-epoch decoding."""
    device = device if device is not None else pick_device()

    def prepare(train):
        images, labels = load_mnist_split(train, image_size)
        if normalize:
            images = normalize_images(images, image_size)
        return images.to(device), labels.to(device)

    train_images, train_labels = prepare(True)
    test_images, test_labels = prepare(False)

    return (
        TensorBatches(train_images, train_labels, batch_size, shuffle=True, drop_last=drop_last),
        TensorBatches(test_images, test_labels, batch_size, shuffle=True, drop_last=drop_last),
    )
