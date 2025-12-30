import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW


class NERLightningModule(LightningModule):
    def __init__(self, model, num_labels, loss_fn, batch_size, epochs, learning_rate):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        self.model = model
        self.loss_fn = loss_fn
        self.num_labels = num_labels
        
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def _compute_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = ((predictions == labels) & mask).sum().item()
        total = mask.sum().item()
        return correct / total if total > 0 else 0.0

    def _shared_step(self, batch, stage):
        labels = batch['labels'].long()
        input_ids = batch['input_ids'].long()
        attention_mask = batch['attention_mask'].long()
        
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        accuracy = self._compute_accuracy(logits, labels)
        
        self.log(f'{stage}_loss', loss, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log(f'{stage}_acc', accuracy, prog_bar=True, batch_size=self.hparams.batch_size)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'valid')
