import torch
import argparse
from phase2.model import Siamese_SBIR
from phase2.train import train_model, get_heats_map
from phase2.inferences import inference_model


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
    parsers.add_argument('--inference', type=bool, default=False)
    parsers.add_argument('--heats_map', type=bool, default=False)
    
    parsers.add_argument('--batch_size', type=int, default=48)
    parsers.add_argument('--test_batch_size', type=int, default=1)
    parsers.add_argument('--steps', type=int, default=20)
    parsers.add_argument('--gamma', type=float, default=0.5)
    parsers.add_argument('--margin', type=float, default=0.3)
    parsers.add_argument('--alpha', type=float, default=0.4)
    parsers.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parsers.add_argument('--threads', type=int, default=4)
    parsers.add_argument('--lr', type=float, default=0.0001)
    parsers.add_argument('--epochs', type=int, default=200)
    
    args = parsers.parse_args()
    model = Siamese_SBIR(args).to(device)
    
    if args.heats_map:
        inference_model(model, args)
        
    if args.inference:
        inference_model(model, args)

    else:
        if args.load_pretrained is False:
            backbones_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_top5_backbone.pth", weights_only=True)
            attention_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_top5_attention.pth", weights_only=True)
            linear_state = torch.load(args.pretrained_dir + "/" + args.dataset_name + "_top5_linear.pth", weights_only=True)

            model.sample_embedding_network.load_state_dict(backbones_state['sample_embedding_network'], strict=False)
            model.attention.load_state_dict(attention_state['attention'], strict=False)
            model.linear.load_state_dict(linear_state['linear'])
            model.sketch_embedding_network.load_state_dict(backbones_state['sketch_embedding_network'], strict=False)
            model.sketch_attention.load_state_dict(attention_state['sketch_attention'], strict=False)
            model.attn.proj.load_state_dict(linear_state['sketch_linear'], strict=False)
        
        train_model(model, args)