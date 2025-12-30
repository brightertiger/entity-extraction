import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data import NERDataset, NERDataModule
from src.model import NERModel
from src.score import evaluate_model
from src.train import NERLightningModule


class Pipeline(ABC):
    def __init__(self, config):
        self.config = config
        
    @abstractmethod
    def run(self):
        pass


class DataPipeline(Pipeline):
    def run(self):
        path = Path(self.config.data.path)
        path.mkdir(parents=True, exist_ok=True)
        
        data = pd.read_csv(path / 'data.csv', encoding_errors='ignore')
        data = data[['Sentence #', 'Word', 'Tag']]
        data.columns = ['index', 'word', 'token']
        data['index'] = data['index'].notna().astype(int).cumsum()
        data = data.reset_index(drop=True)
        
        unique_tokens = np.unique(data['token'])
        mapping = {val: idx for idx, val in enumerate(reversed(unique_tokens))}
        
        train_mask = data['index'] <= 40000
        train = data[train_mask].copy().reset_index(drop=True)
        valid = data[~train_mask].copy().reset_index(drop=True)
        
        train['index'] = train['index'] - train['index'].min()
        valid['index'] = valid['index'] - valid['index'].min()
        
        train.to_csv(path / 'train.csv', index=False)
        valid.to_csv(path / 'valid.csv', index=False)
        
        with open(path / 'mapping.json', 'w') as f:
            json.dump(mapping, f, indent=4)
            
        print(f'✓ Data preparation complete - files saved to {path}')


class TrainPipeline(Pipeline):
    def run(self):
        path = Path(self.config.data.path)
        model_name = self.config.params.hfmodel
        device = self.config.params.device
        batch_size = int(self.config.params.bsize)
        learning_rate = float(self.config.params.learning_rate)
        epochs = int(self.config.params.epochs)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        with open(path / 'mapping.json') as f:
            mapping = json.load(f)

        data_module = NERDataModule(
            train_path=path / 'train.csv',
            valid_path=path / 'valid.csv',
            mapping=mapping,
            tokenizer=tokenizer,
            batch_size=batch_size
        )

        model = NERModel(model_name, len(mapping))
        lightning_model = NERLightningModule(
            model=model,
            num_labels=len(mapping),
            loss_fn=nn.CrossEntropyLoss(reduction='mean'),
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )

        model_dir = path / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                dirpath=str(model_dir),
                filename='model.pt-v1.ckpt',
                save_weights_only=True
            )
        ]

        accelerator = 'gpu' if 'cuda' in device else 'cpu'
        trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=epochs,
            logger=WandbLogger(project=self.config.data.wandb),
            callbacks=callbacks
        )
        
        trainer.fit(lightning_model, data_module)
        print(f'✓ Training complete - model saved to {model_dir}')


class EvalPipeline(Pipeline):
    def run(self):
        path = Path(self.config.data.path)
        model_name = self.config.params.hfmodel
        device = self.config.params.device
        batch_size = int(self.config.params.bsize)

        with open(path / 'mapping.json') as f:
            mapping = json.load(f)
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = NERModel(model_name, len(mapping))
        model_path = path / 'model' / self.config.score.model
        
        state_dict = torch.load(model_path, map_location='cpu')['state_dict']
        state_dict = {k.replace('model.encoder.', 'encoder.'): v for k, v in state_dict.items()}
        state_dict = {k.replace('model.classifier.', 'classifier.'): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        valid_data = pd.read_csv(path / 'valid.csv')
        valid_data['word'] = valid_data['word'].fillna('none')
        
        dataset = NERDataset(valid_data, mapping, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        output, report = evaluate_model(model, dataloader, device, batch_size, tokenizer, mapping)
        
        output.to_csv(path / 'score.csv', index=False)
        
        with open(path / 'report.txt', 'w') as f:
            f.write(report)
            
        print(f'✓ Evaluation complete - results saved to {path}')


PIPELINE_REGISTRY = {
    'prepare-data': DataPipeline,
    'train-model': TrainPipeline,
    'score-model': EvalPipeline,
}


def get_pipeline(config, mode):
    pipeline_cls = PIPELINE_REGISTRY.get(mode)
    if pipeline_cls is None:
        raise ValueError(f'Unknown mode: {mode}. Available: {list(PIPELINE_REGISTRY.keys())}')
    return pipeline_cls(config)
