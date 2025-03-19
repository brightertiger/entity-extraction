import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def score_model(model, data, device, bsize, tokenizer, mapping):
    """
    Score the model on validation data
    
    Args:
        model: The NER model
        data: DataLoader with validation data
        device: Device to run inference on
        bsize: Batch size
        tokenizer: Tokenizer for decoding tokens
        mapping: Mapping from IDs to entity tags
        
    Returns:
        output: DataFrame with predictions
        report: Classification report
    """
    model = model.to(device)
    model.eval()
    scores = []
    tokens = []
    labels = []
    
    # Create reverse mapping for decoding
    rev_mapping = {v: k for k, v in mapping.items()}
    
    with tqdm(total=len(data) * bsize, desc="Scoring") as tq:
        with torch.no_grad():
            for sample in data:
                label = sample.pop('labels').long().to(device)
                words = sample.pop('words').long().to(device)
                attns = sample.pop('attns').long().to(device)
                
                preds = model(words, attns)
                preds = torch.softmax(preds, dim=-1)
                
                scores.append(preds.cpu().data.numpy())
                tokens.append(words.cpu().data.numpy())
                labels.append(label.cpu().data.numpy())
                
                tq.update(bsize)
    
    # Process results
    scores = np.vstack(scores).argmax(axis=-1).reshape(-1)
    tokens = np.vstack(tokens).reshape(-1)
    labels = np.vstack(labels).reshape(-1)
    
    output = pd.DataFrame(np.vstack([tokens, labels, scores])).T
    output.columns = ['word', 'label', 'score']
    
    # Add index column to group by sentences
    index = list(range(len(data) * bsize))
    index = np.repeat(np.array(index), 256, -1)  # 256 is max_len
    output.insert(0, 'index', index)
    
    # Filter out padding tokens (-100)
    output = output[output['label'] != -100].copy().reset_index(drop=True)
    
    # Map numeric values back to entity tags
    output['label'] = output['label'].map(lambda x: rev_mapping[x])
    output['score'] = output['score'].map(lambda x: rev_mapping[x])
    
    # Decode token IDs to text
    output['word'] = tokenizer.convert_ids_to_tokens(output['word'].tolist())
    
    # Prepare for seqeval evaluation
    labels = output.groupby('index')['label'].apply(list).tolist()
    score = output.groupby('index')['score'].apply(list).tolist()
    report = classification_report(labels, score)
    
    return output, report
