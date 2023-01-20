import sys
import json
import wandb
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from transformers import AutoTokenizer
from source.data import ValidDataset, DataModule
from source.model import Model
from source.train import LightModel
from source.score import score_model
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint 
from omegaconf import OmegaConf

CONFIG = OmegaConf.load('./conf.yaml')
parser = argparse.ArgumentParser()
choices = ['prepare-data', 'train-model', 'score-model']
parser.add_argument('-m', '--mode', dest='mode', choices=choices, help="Run Mode")
args = parser.parse_args()

if args.mode == 'prepare-data':
    
    path = CONFIG['data']['path']
    
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

if args.mode == 'train-model':
    
    path = CONFIG['data']['path']
    wandb = CONFIG['data']['wandb']
    hfmodel = CONFIG['params']['hfmodel']
    device = CONFIG['params']['device']
    bsize = int(CONFIG['params']['bsize'])
    learning_rate = float(CONFIG['params']['learning_rate'])
    epochs = int(CONFIG['params']['epochs'])
    tokenizer = AutoTokenizer.from_pretrained(hfmodel)
    mapping = json.load(open('./data/mapping.json'))

    train_data = path + '/train.csv'
    valid_data = path + '/valid.csv'
    data = DataModule(train_data, valid_data, mapping, tokenizer, bsize)

    model = Model(hfmodel, len(mapping))
    model = LightModel(model, len(mapping), nn.CrossEntropyLoss(reduction='mean'), bsize, epochs, learning_rate)

    callbacks = []
    callbacks += [LearningRateMonitor(logging_interval="step")]
    callbacks += [ModelCheckpoint(dirpath=path+'/model', filename='model.pt-v1.ckpt', save_weights_only=True)]

    trainer = {}
    trainer['accelerator'] = 'gpu' if 'cuda' in device else 'cpu'
    trainer['max_epochs'] = epochs
    trainer['logger'] = WandbLogger(project=wandb)
    trainer['callbacks'] = callbacks
    trainer = pl.Trainer(**trainer)
    trainer.fit(model, data)

if args.mode == 'score-model':

    path = CONFIG['data']['path']
    hfmodel = CONFIG['params']['hfmodel']
    device = CONFIG['params']['device']
    bsize = CONFIG['params']['bsize']

    mapping = json.load(open(path + '/mapping.json'))
    tokenizer = AutoTokenizer.from_pretrained(hfmodel)

    model = Model(hfmodel, len(mapping))
    weights = torch.load(path + 'model/' + CONFIG['score']['model'], map_location='cpu')['state_dict']
    weights = {key.replace('model.model.','model.'): val for key, val in weights.items()}
    weights = {key.replace('model.head.','head.'): val for key, val in weights.items()}
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()

    score_data = pd.read_csv(path + '/valid.csv')
    score_data['word'] = score_data['word'].fillna("none")
    score_data = ValidDataset(score_data, mapping, tokenizer)
    score_data = DataLoader(score_data, batch_size=bsize, shuffle=False, drop_last=True)
    
    output, report = score_model(model, score_data, device, bsize, tokenizer, mapping)
    output.to_csv(path + 'score.csv', index=False)
    
    with open(path + '/report.txt','w') as file:
        file.write(report)
    file.close()
