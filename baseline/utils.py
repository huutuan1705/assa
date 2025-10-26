import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def info_nce_loss(args, features_view1: torch.Tensor, features_view2: torch.Tensor):
    """
    InfoNCE (NT-Xent) for SimCLR
    features_view1, features_view2: (B, D)
    """
    temperature = float(args.temperature)
    B, D = features_view1.shape
    device = features_view1.device

    z = torch.cat([features_view1, features_view2], dim=0)

    logits = z @ z.t()                              # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask, float('-inf'))

    logits = logits / temperature

    labels = torch.cat([
        torch.arange(B, 2*B, device=device),
        torch.arange(0, B, device=device)
    ], dim=0).long()

    loss = F.cross_entropy(logits, labels)
    return loss
    
    
def loss_fn(args, features):
    sketch_list, positive_list, negative_list = [], [], []
    
    for i in range(1, args.num_views + 1):
        sketch_list.append(features[f'sketch_feature_{i}'])
        positive_list.append(features[f'positive_feature_{i}'])
        negative_list.append(features[f'negative_feature_{i}'])
        
    criterion = nn.TripletMarginLoss(margin=args.margin)
    
    sum_sketch_features = torch.cat(sketch_list, dim=0)
    sum_positive_features = torch.cat(positive_list, dim=0)
    sum_negative_feature = torch.cat(negative_list, dim=0)
    
    infonce_cross = info_nce_loss(args=args, features_view1=sum_sketch_features, features_view2=sum_positive_features)
    triplet_loss = criterion(sum_sketch_features, sum_positive_features, sum_negative_feature)
    
    total_loss = args.alpha*triplet_loss 
    if args.use_info:
        total_loss = total_loss + (1 - args.alpha)*infonce_cross   
    return total_loss
    
def get_transform(type, aug_mode=1):
    """
    Get transform for SimCLR with alternating augmentation modes
    """
    strong_color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3)
    if type == 'train':
        if aug_mode == 1:
            # Focus on shaped augmentation
            transform_list = [
                transforms.RandomResizedCrop(299, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            
        elif aug_mode == 2:
            # Focus on color augmentation
            transform_list = [
                transforms.Resize(299), 
                transforms.RandomApply([strong_color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.5),  # Add grayscale
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.8),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
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
