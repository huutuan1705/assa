import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.backbones import InceptionV3
from baseline.attention import Linear_global, SelfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SketchAttention(nn.Module):
    def __init__(self, args):
        super(SketchAttention, self).__init__()
        self.pool_method =  nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(2048)
        self.mha = nn.MultiheadAttention(2048, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        
        self.proj = Linear_global(feature_num=args.output_size)
        
    def forward(self, x, return_attn=False):
        identify = x
        x_att = self.norm(x)
        att_out, att_w  = self.mha(x_att, x_att, x_att)
        att_out = self.dropout(att_out)
        attn = identify * att_out + identify
        attn = F.normalize(attn)
        
        output = self.proj(attn)
        if return_attn:
            return output, att_w
        return output

class Siamese_SBIR(nn.Module):
    def __init__(self, args):
        super(Siamese_SBIR, self).__init__()
        self.args = args
        self.sample_embedding_network = InceptionV3(args=args)
        self.attention = SelfAttention(args)
        self.linear = Linear_global(feature_num=args.output_size)

        self.sketch_embedding_network = InceptionV3(args=args)
        self.sketch_attention = SelfAttention(args)
        self.sketch_linear = Linear_global(feature_num=args.output_size)
        
        self.sample_embedding_network.fix_weights()
        self.sketch_embedding_network.fix_weights()
        self.attention.fix_weights()
        self.sketch_attention.fix_weights()
        self.linear.fix_weights()
        
        self.attn = SketchAttention(args)        

if __name__ == "__main__":
    dim = 2048
    dt_rank = 4
    dim_inner = 32
    d_state = 8