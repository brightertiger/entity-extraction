import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

def score_model(model, data, device, bsize, tokenizer, mapping):
    model = model.to(device)
    model.eval()
    scores = []
    tokens = []
    labels = []
    tq = tqdm(total=len(data) * bsize, disable=False)
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
    tq.close()
    scores = np.vstack(scores).argmax(axis=-1).reshape(-1)
    tokens = np.vstack(tokens).reshape(-1)
    labels = np.vstack(labels).reshape(-1)
    output = pd.DataFrame(np.vstack([tokens, labels, scores])).T
    output.columns = ['word', 'label','score']
    index = list(range(len(data) * bsize))
    index = np.repeat(np.array(index), 256, -1)
    output.insert(0, 'index', index)
    output = output[output['label'] != -100].copy().reset_index(drop=True)
    output['label'] = output['label'].map(lambda x : {v:k for k,v in mapping.items()}[x])
    output['score'] = output['score'].map(lambda x : {v:k for k,v in mapping.items()}[x])
    output['word'] = tokenizer.convert_ids_to_tokens(output['word'].tolist())
    labels = output.groupby('index')['label'].apply(list).tolist()
    score = output.groupby('index')['score'].apply(list).tolist()
    report = classification_report(labels, score)
    return output, report
