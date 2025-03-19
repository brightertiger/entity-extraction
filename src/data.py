import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    """Base dataset class for NER tasks"""
    
    def __init__(self, data, mapping, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.max_len = max_len
        
    def __len__(self):
        return self.data['index'].max() - self.data['index'].min() + 1

    def __tokenize__(self, words, labels):
        words_, labels_ = [], []
        for word, label in zip(words, labels):
            word = self.tokenizer.encode(word, add_special_tokens=False)
            label = [label] * len(word)
            words_.extend(word)
            labels_.extend(label)
        return words_, labels_
    
    def __pad__(self, words, labels):
        words = words[:self.max_len-2]
        labels = labels[:self.max_len-2]
        words = [self.tokenizer.cls_token_id] + words + [self.tokenizer.sep_token_id]
        labels = [self.mapping['O']] + [self.mapping[x] for x in labels] + [self.mapping['O']] 
        if len(words) <= self.max_len:
            npad = self.max_len - len(words)
            words.extend([self.tokenizer.pad_token_id] * npad)
            labels.extend([-100] * npad)
        return words, labels

    def __getitem__(self, index):    
        subset = self.data[self.data['index'] == index].reset_index(drop=True)
        words, labels = subset['word'].tolist(), subset['token'].tolist()
        words, labels = self.__tokenize__(words, labels)
        words, labels = self.__pad__(words, labels)
        words = torch.tensor(words).reshape(-1,).long()
        labels = torch.tensor(labels).reshape(-1,).long()
        attns = (words != self.tokenizer.pad_token_id).long()
        return {'words': words, 'labels': labels, 'attns': attns}


class TrainDataset(BaseDataset):
    """Dataset for training examples"""
    pass


class ValidDataset(BaseDataset):
    """Dataset for validation and testing examples"""
    pass
    

class DataModule(pl.LightningDataModule):
    """Lightning data module for training and validation"""
    
    def __init__(self, train_data, valid_data, mapping, tokenizer, bsize):
        super().__init__()
        self.train_data = pd.read_csv(train_data)
        self.valid_data = pd.read_csv(valid_data)
        self.train_data['word'] = self.train_data['word'].fillna("none")
        self.valid_data['word'] = self.valid_data['word'].fillna("none")
        self.mapping = mapping
        self.tokenizer = tokenizer
        self.bsize = bsize

    def train_dataloader(self):
        train_data = TrainDataset(self.train_data, self.mapping, self.tokenizer)
        train_data = DataLoader(train_data, shuffle=True, batch_size=self.bsize, drop_last=True)
        return train_data

    def val_dataloader(self):
        valid_data = ValidDataset(self.valid_data, self.mapping, self.tokenizer)
        valid_data = DataLoader(valid_data, shuffle=False, batch_size=self.bsize, drop_last=True)
        return valid_data

