import torch
import torch.nn as nn 
from transformers import AutoModel

class Model(nn.Module):

    def __init__(self, hfmodel, tags):
        super().__init__()
        self.model = AutoModel.from_pretrained(hfmodel)
        self.head = nn.Linear(768, tags)
        return None

    def forward(self, inputs, mask):
        embed = self.model(input_ids=inputs, attention_mask=mask)
        embed = embed['last_hidden_state']
        output = self.head(embed)
        return output