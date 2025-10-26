import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def get_transform(type, aug_mode='geometric_strong'):
    """
    Get transform for SimCLR with alternating augmentation modes
    
    Args:
        type: 'train' or 'val'/'test'
        aug_mode: 'default', 'geometric_strong', 'color_strong'
            - 'default': balanced augmentation
            - 'geometric_strong': strong geometric, weak color
            - 'color_strong': strong color, weak geometric
    """
    weak_color_jitter = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05)
    strong_color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    if type == 'train':
        if aug_mode == 'geometric_strong':
            # Strong geometric augmentation, weak color augmentation
            transform_list = [
                transforms.RandomResizedCrop(299, scale=(0.7, 1.0)),  # Stronger crop
                transforms.RandomHorizontalFlip(0.7),  # Higher flip probability
                transforms.RandomRotation(50),  # Stronger rotation
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Add affine
                # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),  # Weak color
                transforms.RandomApply([weak_color_jitter], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
                # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            
        elif aug_mode == 'color_strong':
            # Strong color augmentation, weak geometric augmentation
            transform_list = [
                transforms.Resize(299),  # Weaker crop
                # transforms.RandomHorizontalFlip(0.5),  # Lower flip probability
                transforms.RandomRotation(5),  # Weaker rotation
                # transforms.ColorJitter(),  # Strong color
                transforms.RandomApply([strong_color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.5),  # Add grayscale
                # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.8),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            
        else:  # default mode
            # Balanced augmentation (original)
            transform_list = [
                transforms.RandomResizedCrop(299, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        
    else:  # validation/test
        transform_list = [
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    return transforms.Compose(transform_list)
