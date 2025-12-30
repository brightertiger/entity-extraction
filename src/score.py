import numpy as np
import pandas as pd
import torch
from seqeval.metrics import classification_report
from tqdm import tqdm


def evaluate_model(model, dataloader, device, batch_size, tokenizer, mapping):
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_tokens = []
    all_labels = []
    
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            labels = batch['labels'].long().to(device)
            input_ids = batch['input_ids'].long().to(device)
            attention_mask = batch['attention_mask'].long().to(device)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.softmax(logits, dim=-1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_tokens.append(input_ids.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    predictions = np.vstack(all_predictions).argmax(axis=-1).reshape(-1)
    tokens = np.vstack(all_tokens).reshape(-1)
    labels = np.vstack(all_labels).reshape(-1)
    
    output = pd.DataFrame({
        'word': tokens,
        'label': labels,
        'prediction': predictions
    })
    
    max_len = 256
    num_samples = len(dataloader) * batch_size
    sentence_indices = np.repeat(np.arange(num_samples), max_len)
    output.insert(0, 'index', sentence_indices)
    
    output = output[output['label'] != -100].copy().reset_index(drop=True)
    
    output['label'] = output['label'].map(reverse_mapping)
    output['prediction'] = output['prediction'].map(reverse_mapping)
    output['word'] = tokenizer.convert_ids_to_tokens(output['word'].tolist())
    
    grouped_labels = output.groupby('index')['label'].apply(list).tolist()
    grouped_predictions = output.groupby('index')['prediction'].apply(list).tolist()
    
    report = classification_report(grouped_labels, grouped_predictions)
    
    return output, report
