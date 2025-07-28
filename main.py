$ cat main.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from transformers import BertTokenizerFast, BertModel
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

df = pd.read_csv("./archive/phishing_email.csv")
print(df.label.value_counts())

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)
MAX_LEN = 128

def encode_texts(texts):
    """
    Encode a list of texts into token IDs and attention masks for BERT.

    Args:
        texts (list or pd.Series): List of text strings to encode.

    Returns:
        dict: A dictionary containing 'input_ids' and 'attention_mask' tensors.
    """
    with torch.no_grad():
        return tokenizer(
            list(texts),
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )

class PhishingEmailClassifier(nn.Module):
    """
    A BERT-based classifier for phishing email detection.
    """
    def __init__(self, unfreeze_bert: bool = False):
        """
        Initialize the classifier model.

        Args:
            unfreeze_bert (bool): If False, freeze BERT weights during training.
        """
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if not unfreeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.

        Returns:
            torch.Tensor: Logits for the binary classification.
        """
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_token))

def train_model(model, train_loader, val_loader, lr=2e-5, epochs=2, patience=2, fold=0):
    """
    Train the classifier model with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        lr (float): Learning rate.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping.
        fold (int): Fold number for logging.

    Returns:
        nn.Module: The trained model with best validation performance.
    """
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    patience_counter = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = f"best_model_fold_{fold}_{timestamp}.pt"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask).view(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                logits = model(input_ids, attention_mask).view(-1)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                preds = torch.sigmoid(logits) > 0.5
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(torch.load(best_model_path))
    return model

def evaluate_model(model, val_loader):
    """
    Evaluate the model's performance on validation data.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for validation data.

    Returns:
        tuple: Contains accuracy, F1 score, precision, recall, and confusion matrix.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            logits = model(input_ids, attention_mask).view(-1)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_preds = (torch.sigmoid(all_logits) > 0.5).int().numpy()
    all_labels = all_labels.int().numpy()

    return (
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds),
        precision_score(all_labels, all_preds),
        recall_score(all_labels, all_preds),
        confusion_matrix(all_labels, all_preds)
    )

# Hyperparameters and setup
batch_sizes = [16, 32]
learning_rates = [2e-5, 3e-5]
EPOCHS = 2
FOLDS = 3

results = []
X = df.text_combined.values
y = df.label.values
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# Initialize tokenizer and base model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)
base_model = BertModel.from_pretrained('bert-base-uncased')

# Encode all data once
encodings = tokenizer(
    list(df['text_combined']),
    padding='max_length',
    truncation=True,
    max_length=MAX_LEN,
    return_tensors='pt'
)
all_input_ids = encodings['input_ids']
all_attention_mask = encodings['attention_mask']
all_labels = torch.tensor(df['label'].values, dtype=torch.float)

# Loop over hyperparameter combinations
for batch_size in batch_sizes:
    for lr in learning_rates:
        print(f"Testing: Batch Size={batch_size}, Learning Rate={lr}")
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            train_dataset = TensorDataset(
                all_input_ids[train_idx],
                all_attention_mask[train_idx],
                all_labels[train_idx]
            )
            val_dataset = TensorDataset(
                all_input_ids[val_idx],
                all_attention_mask[val_idx],
                all_labels[val_idx]
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
