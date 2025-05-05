# from llama_index.core.embeddings.adapter_utils import BaseAdapter
from llama_index.embeddings.adapter import BaseAdapter
from torch.nn import functional as F
from torch import nn, Tensor
from typing import Dict
import torch

        
config_TwoLayerNN = {
    "small-1.5": {
        "in_features": 384, 
        "hidden_features": 1024,
        "out_features": 384,
        "bias": True,
        "add_residual": True
    },
    "base-1.5": {
        "in_features": 768, 
        "hidden_features": 2048,
        "out_features": 768,
        "bias": True,
        "add_residual": True
    },
    "large-1.5": {
        "in_features": 1024, 
        "hidden_features": 3072,
        "out_features": 1024,
        "bias": True,
        "add_residual": True
    },
}

config_LinearLayer = {
    "small-1.5": {
        "in_features": 384, 
        "out_features": 384,
        "bias": True,
    },
    "base-1.5": {
        "in_features": 768, 
        "out_features": 768,
        "bias": True,
    },
    "large-1.5": {
        "in_features": 1024, 
        "out_features": 1024,
        "bias": True,
    },
}


config_SelfAttentionAdapter = {
    "small-1.5": {
        "embedding_dim": 384,
        "n_heads": 4,
    },
    "base-1.5": {
        "embedding_dim": 768,
        "n_heads": 8,
    },
    "large-1.5": {
        "embedding_dim": 1024,
        "n_heads": 16,
    },
}


class TwoLayerNN(BaseAdapter):
    def __init__(
        self,
        in_features = 384,
        hidden_features = 1024,
        out_features = 384,
        bias = True,
        add_residual = True,
    ):
        super(BaseAdapter, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias

        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
        self._add_residual = add_residual
       
        self.residual_weight = nn.Parameter(torch.zeros(1))

    def forward(self, embed):
        output1 = self.linear1(embed)
        output1 = F.relu(output1)
        output2 = self.linear2(output1)

        if self._add_residual:
            output2 = self.residual_weight * output2 + embed

        return output2

    def get_config_dict(self):
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "add_residual": self._add_residual,
        }
        

class SelfAttentionAdapter(BaseAdapter):
    def __init__(self, embedding_dim, n_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        attn_out, _ = self.attention(x, x, x)  # batch_size, seq_len, emb_dim
        output = self.norm(x + attn_out)  # residual + norm
        return output.squeeze()
    
    def get_config_dict(self):
        return {
            "embedding_dim": self.embedding_dim,
            "n_heads": self.n_heads,
        }


