import torch
import torch.nn as nn
import random
import warnings

random.seed(1234)
warnings.filterwarnings("ignore")

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, device, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)  # [batch_size, seq_len, d_model]