import torchvision.transforms as transforms
from torchvision.transforms import AutoAugmentPolicy

class Autoaugment_preprocess:
    def __init__(self, channels, resize_dim, crop_dim, policy=AutoAugmentPolicy.IMAGENET):
        """
        Uses as default AutoAugmentPolicy.IMAGENET, other options are:
        - AutoAugmentPolicy.CIFAR10
        - AutoAugmentPolicy.SVHN
        """
        # ImageNet1k mean and std
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        if channels == 1:
            mean, std = (0.5,), (0.5,)
        self.transform = transforms.Compose([
            transforms.AutoAugment(policy=policy),
            transforms.Resize(resize_dim),                                                   
            transforms.RandomCrop(crop_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])