import torch
import torch.nn as nn

from baseline.backbones import InceptionV3
from baseline.attention import Linear_global, SelfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Siamese_SBIR(nn.Module):
    def __init__(self, args):
        super(Siamese_SBIR, self).__init__()
        self.args = args
        self.sample_embedding_network = InceptionV3(args=args)
        self.attention = SelfAttention(args)
        self.linear = Linear_global(feature_num=self.args.output_size)
        
        self.sketch_embedding_network = InceptionV3(args=args)
        self.sketch_attention = SelfAttention(args)
        self.sketch_linear = Linear_global(feature_num=self.args.output_size)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_normal_(m.weight)
        
        if self.args.use_kaiming_init:
            self.attention.apply(init_weights)
            self.sketch_attention.apply(init_weights)
            
            self.linear.apply(init_weights)
            self.sketch_linear.apply(init_weights)
            
    def extract_feature(self, batch, num):
        sketch_img = batch[f'sketch_img_{num}'].to(device)
        positive_img = batch[f'positive_img_{num}'].to(device)
        negative_img = batch[f'negative_img_{num}'].to(device)
        
        positive_feature = self.sample_embedding_network(positive_img)
        negative_feature = self.sample_embedding_network(negative_img)
        sketch_feature = self.sketch_embedding_network(sketch_img)
        
        positive_feature = self.attention(positive_feature)
        negative_feature = self.attention(negative_feature)
        sketch_feature = self.sketch_attention(sketch_feature)
        
        positive_feature = self.linear(positive_feature)
        negative_feature = self.linear(negative_feature)
        sketch_feature = self.sketch_linear(sketch_feature)
        
        return sketch_feature, positive_feature, negative_feature
    
    def forward(self, batch):
        outputs = {}
        for i in range(1, self.args.num_views+1):
            sketch_feature, positive_feature, negative_feature = self.extract_feature(batch=batch, num=i)
            outputs[f'sketch_feature_{i}']   = sketch_feature
            outputs[f'positive_feature_{i}'] = positive_feature
            outputs[f'negative_feature_{i}'] = negative_feature
            
        return outputs