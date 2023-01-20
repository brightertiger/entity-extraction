import sys
sys.path.append('..')

import json
import wandb
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

mode = 'train-model'

if mode == 'prepare-data':
    
    path = './data/'
    
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


if mode == 'train-model':
    
    bsize = 12
    path = './data/'
    hfmodel = 'roberta-base'
    wandb = 'entity-extraction'
    device = 'cuda:0'
    
    tokenizer = AutoTokenizer.from_pretrained(hfmodel)
    mapping = json.load(open('./data/mapping.json'))

    train_data = path + '/train.csv'
    valid_data = path + '/valid.csv'
    mapping = path + '/mapping.json'
    data = DataModule(train_data, valid_data, mapping, tokenizer, bsize)

    model = Model(hfmodel, len(mapping))
    model = LightModel(model, len(mapping), nn.CrossEntropyLoss(reduction='mean'), 16, 5, 1e-5)

    callbacks = []
    callbacks += [LearningRateMonitor(logging_interval="step")]
    callbacks += [ModelCheckpoint(dirpath=path+'/model', filename='model.pt', save_weights_only=True)]

    trainer = {}
    trainer['accelerator'] = 'gpu' if 'cuda' in device else 'cpu'
    trainer['max_epochs'] = 2
    trainer['logger'] = WandbLogger(project=wandb)
    trainer['callbacks'] = callbacks
    trainer = pl.Trainer(**trainer)
    trainer.fit(model, data)

if mode == 'score-model':

    path = './data/'
    hfmodel = 'roberta-base'
    device = 'cuda:0'
    bsize = 12

    mapping = json.load(open(path + '/mapping.json'))
    tokenizer = AutoTokenizer.from_pretrained(hfmodel)

    model = Model(hfmodel, len(mapping))
    weights = torch.load(path + 'model.pt', map_location='cpu')['model_state_dict']
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

    