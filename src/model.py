import torch
import torch.nn as nn 
from transformers import AutoModel

class Model(nn.Module):
    """NER model using pretrained transformer with linear classification head"""

    def __init__(self, hfmodel, tags):
        super().__init__()
        self.model = AutoModel.from_pretrained(hfmodel)
        self.head = nn.Linear(768, tags)
    
    def forward(self, inputs, mask):
        """
        Args:
            inputs: Token ids (batch_size, seq_len)
            mask: Attention mask (batch_size, seq_len)
            
        Returns:
            output: Classification logits (batch_size, seq_len, num_tags)
        """
        embed = self.model(input_ids=inputs, attention_mask=mask)
        embed = embed['last_hidden_state']
        output = self.head(embed)
        return output