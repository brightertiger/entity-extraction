# Entity Extraction

## Overview

This repository contains code for Named-Entity Recognition (NER) using finetuned RoBERTa models. The dataset has been borrowed from [Kaggle](https://www.kaggle.com/datasets/namanj27/ner-dataset). The objective is to accurately predict entity tags for words in sentences, distinguishing between different entity types such as people (per), organizations (org), locations (geo), and more.

## Dataset Structure

The input dataset has the following structure:

|Sentence # |Word         |POS|Tag  |
|-----------|-------------|---|-----|
|Sentence: 1|Thousands    |NNS|O    |
|           |of           |IN |O    |
|           |demonstrators|NNS|O    |
|           |have         |VBP|O    |
|           |marched      |VBN|O    |
|           |through      |IN |O    |
|           |London       |NNP|B-geo|
|           |to           |TO |O    |

Where:
- **Word**: The token from the sentence
- **POS**: Part-of-speech tag
- **Tag**: The entity tag in IOB format (Inside-Outside-Beginning)
  - **B-** prefix indicates the beginning of an entity
  - **I-** prefix indicates the continuation of an entity
  - **O** indicates tokens that are not part of any entity

## ML Approach

### Model Architecture
We use the RoBERTa base model, which is an optimized version of BERT with improved training methodology. RoBERTa features:
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- 125M parameters

For NER, we add a token classification head on top of the RoBERTa encoder, which consists of a dropout layer followed by a linear layer mapping to the number of entity classes.

### Training Process
The model is trained using:
- Cross-entropy loss function
- AdamW optimizer with learning rate of 2e-5
- Linear learning rate scheduler with warmup
- Batch size of 16
- Training runs for 5 epochs
- Gradient accumulation steps of 2
- Max sequence length of 128 tokens

### Data Preprocessing
- Sentences are tokenized using RoBERTa's tokenizer
- Special handling for subword tokenization to maintain entity boundaries
- Entity tags are converted to numerical indices using a mapping dictionary
- Dataset is split into training (80%) and validation (20%) sets

### Evaluation Metrics
We evaluate the model using:
- **Precision**: Ratio of correctly predicted entities to all predicted entities
  - Formula: TP / (TP + FP)
  - Measures the accuracy of positive predictions
  - Higher precision means fewer false entity identifications
  
- **Recall**: Ratio of correctly predicted entities to all actual entities
  - Formula: TP / (TP + FN)
  - Measures the model's ability to find all entities
  - Higher recall means fewer missed entities
  
- **F1 Score**: Harmonic mean of precision and recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
  - Balances the trade-off between precision and recall
  - Critical for NER tasks where both false positives and false negatives are problematic
  
- **Support**: Number of occurrences of each entity type
  - Helps interpret metrics in light of class imbalance
  - Low-support classes (like 'art' and 'eve' in our dataset) typically have less reliable metrics

We report metrics at multiple levels:
- **Per-entity metrics**: Performance on each entity type (art, geo, per, etc.)
- **Micro average**: Calculated by aggregating contributions of all classes
  - Favors performance on majority classes
- **Macro average**: Simple average of per-class metrics
  - Each class contributes equally regardless of support
  - Lower macro vs. micro indicates poorer performance on minority classes
- **Weighted average**: Average weighted by the support of each class
  - Reflects overall performance while accounting for class frequency

For NER specifically, we use a strict match evaluation - an entity prediction is considered correct only if both the entity type and its exact boundary (start and end positions) match the ground truth.

## Code Structure

```
├── data                        # DATA FILES
│   ├── data.csv                    # Raw Dataset from Kaggle 
│   ├── train.csv                   # Training split from data.csv
│   └── valid.csv                   # Validation split from data.csv
│   ├── mapping.json                # Mapping of labels to index
│   ├── score.csv                   # Model predictions on valid.csv
│   ├── model                       # Finetuned Model
│   │   └── model.pt-v1.ckpt
│   ├── report.txt                  # Evaluation on valid.csv

├── source                      # SOURCE CODE
│   ├── data.py                     # Data Loaders
│   ├── model.py                    # HuggingFace Model
│   ├── score.py                    # Scoring Model
│   └── train.py                    # Training Model

├── main.py                     
├── conf.yaml                   

```

### Instructions

- Download the "data.csv" file from Kaggle and place it in data folder. 

- Command to split to preprocess data file and split it into training, validation

```shell
python main.py --mode prepare-data
```

- Command to finetune the model

```shell
python main.py --mode train-model
```

- Command to evaluate the model
```shell
python main.py --mode score-model
```


### Performance

The final model performance is saved in report.txt file. It looks like:

|label |precision| recall |f1-score|support|
|------|---------|--------|--------|------ |
|art          |0.38  |0.06    |0.10   |161    |      
|eve          |0.22  |0.23    |0.23   |61     |      
|geo          |0.86  |0.88    |0.87   |12744  |      
|gpe          |0.94  |0.95    |0.95   |5020   |      
|nat          |0.35  |0.44    |0.39   |73     |      
|org          |0.70  |0.70    |0.70   |6655   |      
|per          |0.76  |0.81    |0.78   |5195   |      
|tim          |0.83  |0.82    |0.82   |3942   |
|             |      |        |       |       |
|micro avg    |0.82  |0.83    |0.82   |33851  |
|macro avg    |0.63    |0.61   |0.60  |33851  |
|weighted avg |0.82  |0.83    |0.82   |33851  |  

#### Performance Analysis

The performance metrics reveal several important insights:

1. **Class imbalance impact**: 
   - High-support classes (geo: 12,744, org: 6,655, per: 5,195) achieve F1 scores of 0.87, 0.70, and 0.78 respectively
   - Low-support classes (art: 161, eve: 61, nat: 73) achieve much lower F1 scores of 0.10, 0.23, and 0.39

2. **Precision-recall balance**:
   - For most entity types, precision and recall are fairly balanced
   - Exception is 'art' entities where precision (0.38) significantly exceeds recall (0.06), indicating the model is conservative in predicting this class

3. **Overall performance**:
   - Micro-average F1 of 0.82 indicates strong general performance
   - Gap between micro-average (0.82) and macro-average (0.60) F1 scores confirms the model performs substantially better on common entity types

4. **Areas for improvement**:
   - Targeted strategies for low-resource classes:
     - Data augmentation for minority classes
     - Class weighting in loss function
     - Few-shot or meta-learning approaches
   - Entity boundary detection could be improved for complex entities

The weighted average metrics closely match the micro-average, confirming that performance is dominated by high-frequency classes.

The wandb logs are below:

![img](data/log.png)


