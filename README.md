# Entity Extraction

Named Entity Recognition (NER) using fine-tuned RoBERTa models. The dataset is from [Kaggle NER Dataset](https://www.kaggle.com/datasets/namanj27/ner-dataset).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data (place data.csv from Kaggle in data/ folder first)
python main.py --mode prepare-data

# Train model
python main.py --mode train-model

# Evaluate model
python main.py --mode score-model
```

## Dataset Structure

| Sentence # | Word | POS | Tag |
|------------|------|-----|-----|
| Sentence: 1 | Thousands | NNS | O |
| | of | IN | O |
| | demonstrators | NNS | O |
| | London | NNP | B-geo |

**Tag Format (IOB):**
- `B-` prefix: Beginning of an entity
- `I-` prefix: Inside/continuation of an entity
- `O`: Outside any entity

**Entity Types:** geo (geographical), gpe (geopolitical), org (organization), per (person), tim (time), art (artifact), eve (event), nat (natural phenomenon)

## Model Architecture

- **Base Model:** RoBERTa-base (12 layers, 768 hidden, 125M parameters)
- **Classification Head:** Linear layer mapping to entity classes
- **Loss:** Cross-entropy
- **Optimizer:** AdamW

## Configuration

Edit `conf.yaml` to modify:

```yaml
data:
  path: './data/'
  wandb: 'entity-extraction'

params:
  device: 'cuda:0'
  hfmodel: 'roberta-base'
  bsize: 12
  learning_rate: 1e-5
  epochs: 2

score:
  model: 'model.pt-v1.ckpt'
```

## Project Structure

```
├── data/
│   ├── data.csv          # Raw dataset (download from Kaggle)
│   ├── train.csv         # Training split
│   ├── valid.csv         # Validation split
│   ├── mapping.json      # Label to index mapping
│   ├── score.csv         # Model predictions
│   ├── report.txt        # Evaluation metrics
│   └── model/
│       └── model.pt-v1.ckpt
├── src/
│   ├── data.py           # Dataset and DataModule
│   ├── model.py          # NER model
│   ├── pipeline.py       # Training/evaluation pipelines
│   ├── score.py          # Model evaluation
│   └── train.py          # Lightning training module
├── main.py               # Entry point
├── conf.yaml             # Configuration
└── requirements.txt
```

## Performance

| Entity | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| art | 0.38 | 0.06 | 0.10 | 161 |
| eve | 0.22 | 0.23 | 0.23 | 61 |
| geo | 0.86 | 0.88 | 0.87 | 12744 |
| gpe | 0.94 | 0.95 | 0.95 | 5020 |
| nat | 0.35 | 0.44 | 0.39 | 73 |
| org | 0.70 | 0.70 | 0.70 | 6655 |
| per | 0.76 | 0.81 | 0.78 | 5195 |
| tim | 0.83 | 0.82 | 0.82 | 3942 |
| **micro avg** | 0.82 | 0.83 | 0.82 | 33851 |
| **macro avg** | 0.63 | 0.61 | 0.60 | 33851 |

**Key Observations:**
- High-frequency entities (geo, gpe, per) achieve F1 > 0.78
- Low-frequency entities (art, eve, nat) need improvement
- Overall micro-F1 of 0.82 indicates strong general performance

![Training Logs](data/log.png)

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
