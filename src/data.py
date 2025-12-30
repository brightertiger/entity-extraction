import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class NERDataset(Dataset):
    def __init__(self, data, mapping, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.max_len = max_len
        
    def __len__(self):
        return self.data['index'].max() - self.data['index'].min() + 1

    def _tokenize(self, words, labels):
        tokenized_words, tokenized_labels = [], []
        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            tokenized_words.extend(word_tokens)
            tokenized_labels.extend([label] * len(word_tokens))
        return tokenized_words, tokenized_labels
    
    def _pad(self, words, labels):
        words = words[:self.max_len - 2]
        labels = labels[:self.max_len - 2]
        
        words = [self.tokenizer.cls_token_id] + words + [self.tokenizer.sep_token_id]
        labels = [self.mapping['O']] + [self.mapping[x] for x in labels] + [self.mapping['O']]
        
        if len(words) < self.max_len:
            pad_len = self.max_len - len(words)
            words.extend([self.tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
            
        return words, labels

    def __getitem__(self, index):
        subset = self.data[self.data['index'] == index].reset_index(drop=True)
        words = subset['word'].tolist()
        labels = subset['token'].tolist()
        
        words, labels = self._tokenize(words, labels)
        words, labels = self._pad(words, labels)
        
        words = torch.tensor(words, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (words != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': words,
            'labels': labels,
            'attention_mask': attention_mask
        }


class NERDataModule(pl.LightningDataModule):
    def __init__(self, train_path, valid_path, mapping, tokenizer, batch_size, max_len=256):
        super().__init__()
        self.train_data = self._load_data(train_path)
        self.valid_data = self._load_data(valid_path)
        self.mapping = mapping
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    @staticmethod
    def _load_data(path):
        data = pd.read_csv(path)
        data['word'] = data['word'].fillna('none')
        return data

    def train_dataloader(self):
        dataset = NERDataset(self.train_data, self.mapping, self.tokenizer, self.max_len)
        return DataLoader(dataset, shuffle=True, batch_size=self.batch_size, drop_last=True)

    def val_dataloader(self):
        dataset = NERDataset(self.valid_data, self.mapping, self.tokenizer, self.max_len)
        return DataLoader(dataset, shuffle=False, batch_size=self.batch_size, drop_last=True)
