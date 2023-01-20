import torch
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

class LightModel(LightningModule):
    
    def __init__(self, model, nlabels, loss_fn, bsize, epochs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.bsize = bsize
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.nlabels = nlabels
        return None

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}    
    
    def forward(self, words, attns):
        preds = self.model(words, attns)
        return preds

    def training_step(self, batch, batch_idx):
        label = batch.pop('labels').long().unsqueeze(-1)
        words = batch.pop('words').long()
        attns = batch.pop('attns').long()
        preds = self.forward(words, attns)
        loss = self.loss_fn(preds.view(-1, self.nlabels), label.view(-1))
        scores = torch.argmax(preds.squeeze(), dim=-1)
        correct = (scores.cpu().data.numpy() == label.squeeze().cpu().data.numpy()).sum()
        total = attns.reshape(-1).cpu().data.numpy().sum()
        accuracy = correct / total
        self.log(f"train_loss", loss, prog_bar=True, batch_size=self.bsize)
        self.log(f"train_acc", accuracy, prog_bar=True, batch_size=self.bsize)
        return loss

    def validation_step(self, batch, batch_idx):
        label = batch.pop('labels').long().unsqueeze(-1)
        words = batch.pop('words').long()
        attns = batch.pop('attns').long()
        preds = self.forward(words, attns)
        scores = torch.argmax(preds, dim=-1)
        loss = self.loss_fn(preds.view(-1, self.nlabels), label.view(-1))
        scores = torch.argmax(preds.squeeze(), dim=-1)
        correct = (scores.cpu().data.numpy() == label.squeeze().cpu().data.numpy()).sum()
        total = attns.reshape(-1).cpu().data.numpy().sum()
        accuracy = correct / total
        self.log(f"valid_loss", loss, prog_bar=True, batch_size=self.bsize)
        self.log(f"valid_acc", accuracy, prog_bar=True, batch_size=self.bsize)
        return None
    
    