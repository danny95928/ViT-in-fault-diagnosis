#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNormal(nn.Module):
    def __init__(self, hidden_size, esp=1e-6):
        super(LayerNormal, self).__init__()
        self.esp = esp
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True).clamp(min=self.esp)
        out = (x - mu) / sigma
        out = out * self.weight.expand_as(out) + self.bias.expand_as(out)
        return out

class ViTFusionModel(nn.Module):
    def __init__(self, input_dim=256, num_heads=8, num_classes=10):
        super(ViTFusionModel, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = input_dim

        self.biGruModel = BiGruModel(input_size=64, hidden_size=input_dim)
        self.lstmModel = LSTMModel(input_size=64, hidden_size=input_dim)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=input_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        gru_output = self.biGruModel(x)
        lstm_output = self.lstmModel(x)

        combined_features = torch.cat((gru_output, lstm_output), dim=-1)

        transformer_input = combined_features.unsqueeze(0)
        transformer_output = self.transformer_encoder(transformer_input)

        output = transformer_output.mean(dim=0)
        output = self.fc(output)

        return F.log_softmax(output, dim=1)

def get_parameter_number(net, name):
    total_num = sum(p.numel() for p in net.parameters())
    return {'name: {}: ->:{}'.format(name, total_num)}

if __name__ == '__main__':
    model = ViTFusionModel(input_dim=256, num_heads=8, num_classes=10)
    model.eval()
    batch_size = 32
    input_data = torch.randn(batch_size, 16, 64)
    output = model(input_data)
    print(output.size())
    params = get_parameter_number(model, 'ViTFusionModel')
    print(params)

