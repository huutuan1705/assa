import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        self.pool_method =  nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(2048)
        self.mha = nn.MultiheadAttention(2048, num_heads=args.num_heads, batch_first=True)
        # self.mha = nn.MultiheadAttention(2048, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        identify = x
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h*w).transpose(1, 2)
        x_att = self.norm(x_att)
        
        att_out, _  = self.mha(x_att, x_att, x_att)
        att_out = self.dropout(att_out)
        att_out = att_out.transpose(1, 2).reshape(bs, c, h, w)
        
        output = identify * att_out + identify
        output = self.pool_method(output).view(-1, 2048)
        
        return F.normalize(output)
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False

    
class SketchAttention(nn.Module):
    def __init__(self, args):
        super(SketchAttention, self).__init__()
        self.pool_method =  nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(2048)
        # self.mha = nn.MultiheadAttention(2048, num_heads=args.num_heads, batch_first=True)
        self.mha = nn.MultiheadAttention(2048, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        identify = x
        x_att = self.norm(x)
        att_out, _  = self.mha(x_att, x_att, x_att)
        att_out = self.dropout(att_out)
        
        output = identify * att_out + identify
        output = F.normalize(output)
        return output

class Attention_global(nn.Module):
    def __init__(self, args):
        super(Attention_global, self).__init__()
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.net = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(),
                                 nn.Conv2d(512, 1, kernel_size=1))
        
    def forward(self, backbone_tensor):
        backbone_tensor_1 = self.net(backbone_tensor)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), -1)
        backbone_tensor_1 = nn.Softmax(dim=1)(backbone_tensor_1)
        backbone_tensor_1 = backbone_tensor_1.view(backbone_tensor_1.size(0), 1, backbone_tensor.size(2), backbone_tensor.size(3))
        fatt = backbone_tensor*backbone_tensor_1
        fatt1 = backbone_tensor +fatt
        fatt1 = self.pool_method(fatt1).view(-1, 2048)
        return  F.normalize(fatt1)
        
class Linear_global(nn.Module):
    def __init__(self, feature_num, in_features=2048):
        super(Linear_global, self).__init__()
        self.head_layer = nn.Linear(in_features, feature_num)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(x)
        return F.normalize(self.head_layer(x))
    
    def fix_weights(self):
        for x in self.parameters():
            x.requires_grad = False

if __name__ == "__main__":
    dim = 2048
    dt_rank = 4
    dim_inner = 32
    d_state = 8
    
    model = SketchAttention(None)
    x = torch.randn(2, 5, dim)
    out = model(x)
    print("Output shape:", out.shape)