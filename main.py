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
    with torch.no_grad():
        return tokenizer(
            list(texts),
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )

class PhishingEmailClassifier(nn.Module):
    def __init__(self, unfreeze_bert: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if not unfreeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_token))

def train_model(model, train_loader, val_loader, lr=2e-5, epochs=2, patience=2, fold=0):
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

batch_sizes = [16, 32]
learning_rates = [2e-5, 3e-5]
EPOCHS = 2
FOLDS = 3

results = []
X = df.text_combined.values
y = df.label.values
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', use_fast=True)
base_model = BertModel.from_pretrained('bert-base-uncased')

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

            model = PhishingEmailClassifier(unfreeze_bert=False)
            model.bert = base_model
            trained_model = train_model(model, train_loader, val_loader, lr=lr, epochs=EPOCHS, patience=2, fold=fold)

            acc, f1, prec, rec, cm = evaluate_model(trained_model, val_loader)
            print(f" Fold {fold} -- Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
            fold_metrics.append((acc, f1, prec, rec))

        avg_metrics = np.mean(fold_metrics, axis=0)
        results.append({
            "batch_size": batch_size,
            "learning_rate": lr,
            "avg_accuracy": avg_metrics[0],
            "avg_f1": avg_metrics[1],
            "avg_precision": avg_metrics[2],
            "avg_recall": avg_metrics[3],
        })

result_df = pd.DataFrame(results)
print(result_df.sort_values(by="avg_f1", ascending=False))

plt.figure(figsize=(10,6))
sns.barplot(data=result_df, x="batch_size", y="avg_f1", hue="learning_rate")
plt.title("F1 Score across Batch Size and Learning Rate")
plt.show()
