import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from abc import ABC, abstractmethod

from src.data import ValidDataset, DataModule
from src.model import Model
from src.train import LightModel
from src.score import score_model


class Pipeline(ABC):
    """Base pipeline class for NER operations"""
    def __init__(self, config):
        self.config = config
        
    @abstractmethod
    def run(self):
        """Run the pipeline operation"""
        pass


class DataPipeline(Pipeline):
    """Pipeline for data preparation"""
    def run(self):
        path = self.config['data']['path']
        os.makedirs(path, exist_ok=True)
        
        data = pd.read_csv(path + '/data.csv', encoding_errors='ignore')
        data = data[['Sentence #','Word','Tag']]
        data.columns = ['index','word','token']
        data['index'] = data['index'].notna().astype(int)
        data['index'] = data['index'].cumsum()
        data = data.reset_index(drop=True)
        
        mapping = data['token']
        mapping = np.unique(mapping)
        mapping = {val:idx for idx, val in enumerate(reversed(mapping))}
        
        train = data[data['index'] <= 40000].copy().reset_index(drop=True)
        valid = data[data['index'] >  40000].copy().reset_index(drop=True)
        train['index'] = train['index'] - train['index'].min()
        valid['index'] = valid['index'] - valid['index'].min()
        
        train.to_csv(path + '/train.csv', index=False)
        valid.to_csv(path + '/valid.csv', index=False)
        with open(path + '/mapping.json', "w") as outfile:
            json.dump(mapping, outfile, indent=4)
            
        print(f"✓ Data preparation complete - files saved to {path}")


class TrainPipeline(Pipeline):
    """Pipeline for model training"""
    def run(self):
        path = self.config['data']['path']
        wandb_project = self.config['data']['wandb']
        hfmodel = self.config['params']['hfmodel']
        device = self.config['params']['device']
        bsize = int(self.config['params']['bsize'])
        learning_rate = float(self.config['params']['learning_rate'])
        epochs = int(self.config['params']['epochs'])
        
        tokenizer = AutoTokenizer.from_pretrained(hfmodel)
        mapping = json.load(open(os.path.join(path, 'mapping.json')))

        train_data = os.path.join(path, 'train.csv')
        valid_data = os.path.join(path, 'valid.csv')
        data = DataModule(train_data, valid_data, mapping, tokenizer, bsize)

        model = Model(hfmodel, len(mapping))
        model = LightModel(model, len(mapping), nn.CrossEntropyLoss(reduction='mean'), bsize, epochs, learning_rate)

        model_dir = os.path.join(path, 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = []
        callbacks += [LearningRateMonitor(logging_interval="step")]
        callbacks += [ModelCheckpoint(dirpath=model_dir, filename='model.pt-v1.ckpt', save_weights_only=True)]

        trainer_config = {
            'accelerator': 'gpu' if 'cuda' in device else 'cpu',
            'max_epochs': epochs,
            'logger': WandbLogger(project=wandb_project),
            'callbacks': callbacks
        }
        
        trainer = pl.Trainer(**trainer_config)
        trainer.fit(model, data)
        
        print(f"✓ Training complete - model saved to {model_dir}")


class EvalPipeline(Pipeline):
    """Pipeline for model evaluation"""
    def run(self):
        path = self.config['data']['path']
        hfmodel = self.config['params']['hfmodel']
        device = self.config['params']['device']
        bsize = int(self.config['params']['bsize'])

        mapping = json.load(open(os.path.join(path, 'mapping.json')))
        tokenizer = AutoTokenizer.from_pretrained(hfmodel)

        model = Model(hfmodel, len(mapping))
        model_path = os.path.join(path, 'model', self.config['score']['model'])
        
        weights = torch.load(model_path, map_location='cpu')['state_dict']
        weights = {key.replace('model.model.','model.'): val for key, val in weights.items()}
        weights = {key.replace('model.head.','head.'): val for key, val in weights.items()}
        model.load_state_dict(weights)
        model = model.to(device)
        model.eval()

        score_data = pd.read_csv(os.path.join(path, 'valid.csv'))
        score_data['word'] = score_data['word'].fillna("none")
        score_data = ValidDataset(score_data, mapping, tokenizer)
        score_data = DataLoader(score_data, batch_size=bsize, shuffle=False, drop_last=True)
        
        output, report = score_model(model, score_data, device, bsize, tokenizer, mapping)
        output.to_csv(os.path.join(path, 'score.csv'), index=False)
        
        with open(os.path.join(path, 'report.txt'), 'w') as file:
            file.write(report)
            
        print(f"✓ Evaluation complete - results saved to {path}")


def get_pipeline(config, mode):
    """Factory function to get the appropriate pipeline"""
    if mode == 'prepare-data':
        return DataPipeline(config)
    elif mode == 'train-model':
        return TrainPipeline(config)
    elif mode == 'score-model':
        return EvalPipeline(config)
    else:
        raise ValueError(f"Unknown mode: {mode}") 