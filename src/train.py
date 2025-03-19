import torch
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

class LightModel(LightningModule):
    """Lightning module for training NER models"""
    
    def __init__(self, model, nlabels, loss_fn, bsize, epochs, lr):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        self.model = model
        self.loss_fn = loss_fn
        self.nlabels = nlabels
        
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}    
    
    def forward(self, words, attns):
        return self.model(words, attns)

    def _calculate_metrics(self, preds, label, attns):
        scores = torch.argmax(preds, dim=-1)
        mask = (label.squeeze() != -100)  # Ignore padding tokens
        correct = ((scores == label.squeeze()) & mask).sum().item()
        total = mask.sum().item()
        accuracy = correct / total if total > 0 else 0
        return accuracy

    def training_step(self, batch, batch_idx):
        label = batch.pop('labels').long()
        words = batch.pop('words').long()
        attns = batch.pop('attns').long()
        
        preds = self.forward(words, attns)
        loss = self.loss_fn(preds.view(-1, self.nlabels), label.view(-1))
        
        accuracy = self._calculate_metrics(preds, label, attns)
        
        self.log("train_loss", loss, prog_bar=True, batch_size=self.hparams.bsize)
        self.log("train_acc", accuracy, prog_bar=True, batch_size=self.hparams.bsize)
        return loss

    def validation_step(self, batch, batch_idx):
        label = batch.pop('labels').long()
        words = batch.pop('words').long()
        attns = batch.pop('attns').long()
        
        preds = self.forward(words, attns)
        loss = self.loss_fn(preds.view(-1, self.nlabels), label.view(-1))
        
        accuracy = self._calculate_metrics(preds, label, attns)
        
        self.log("valid_loss", loss, prog_bar=True, batch_size=self.hparams.bsize)
        self.log("valid_acc", accuracy, prog_bar=True, batch_size=self.hparams.bsize)
        return loss
    
    