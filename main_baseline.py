import torch
import argparse
from baseline.model import Siamese_SBIR
from baseline.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Baseline Fine-Grained SBIR model')
    parsers.add_argument('--dataset_name', type=str, default='ShoeV2')
    parsers.add_argument('--output_size', type=int, default=64)
    parsers.add_argument('--num_heads', type=int, default=8)
    parsers.add_argument('--root_dir', type=str, default='/kaggle/input/fg-sbir-dataset')
    parsers.add_argument('--pretrained_dir', type=str, default='/kaggle/input/base_ae_model/pytorch/default/1/best_model.pth')
    parsers.add_argument('--save_dir', type=str, default='/kaggle/working/')
    
    parsers.add_argument('--use_kaiming_init', type=bool, default=True)
    parsers.add_argument('--load_pretrained', type=bool, default=False)
    parsers.add_argument('--use_info', type=bool, default=False)
    
    parsers.add_argument('--batch_size', type=int, default=24)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--step_size', type=int, default=100)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--alpha', type=float, default=1)
    parsers.add_argument('--beta', type=float, default=0.2)
    parsers.add_argument('--num_views', type=int, default=2, help='view 1 - 5')
    parsers.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.0001)
    parsers.add_argument('--epochs', type=int, default=200)
    
    args = parsers.parse_args()
    model = Siamese_SBIR(args).to(device)
    train_model(model, args)