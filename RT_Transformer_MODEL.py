# ----------------------------
# Imports
# ----------------------------
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             precision_recall_curve, roc_curve, auc)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
from google.colab import files, drive
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# SECTION 1: Data Preprocessing
# ----------------------------
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/My Drive/Data/bnpl_credit_data_500.csv')

data.fillna(0, inplace=True)
data['Age Condition'] = np.where(data['Age'] < 18, 0, 1)
data['Credit_Condition'] = np.where(data['Credit Score'] > 519, 1, 0)

freq_cols = [f'Monthly Purchase Frequency {i}' for i in range(1, 7)]
amount_cols = [f'Monthly Purchase Amount {i}' for i in range(1, 7)]
data['Total_Purchase_Frequency'] = data[freq_cols].sum(axis=1)
data['Total_Purchase_Amount'] = data[amount_cols].sum(axis=1)
data['Repeat Usage'] = data['Repeat Usage'].map({'Yes': 1, 'No': 0})

def determine_credit(row):
    if row['Credit_Condition'] == 0:
        return 0, 0
    if row['Payment Status'] == 'No':
        if row['Total_Purchase_Amount'] > 310000001:
            return 10000000, 1
        elif row['Total_Purchase_Amount'] > 150000001:
            return 5000000, 1
    else:
        if row['Total_Purchase_Frequency'] > 79 and row['Total_Purchase_Amount'] > 220000000:
            return 10000000, 3
        elif row['Total_Purchase_Frequency'] > 79 and row['Total_Purchase_Amount'] < 220000001:
            return 10000000, 1
        elif row['Total_Purchase_Frequency'] < 80 and row['Total_Purchase_Amount'] > 110000000:
            return 5000000, 3
        elif row['Total_Purchase_Frequency'] < 80 and row['Total_Purchase_Amount'] < 1100000001:
            return 5000000, 1
        elif row['Total_Purchase_Frequency'] < 41 and row['Total_Purchase_Amount'] < 80000001:
            return 2000000, 1
    return 0, 0


data[['Credit Amount', 'Repayment Period']] = data.apply(determine_credit, axis=1, result_type='expand')
data['Target'] = np.where(data['Credit_Condition'] & (data['Total_Purchase_Amount'] > 10), 1, 0)

features = data[['Age', 'Credit Score', 'Total_Purchase_Frequency', 'Total_Purchase_Amount', 'Age Condition', 'Rating', 'Repeat Usage']]
target = data['Target']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.3, random_state=42, stratify=target)

# Downsample Train Data
X_train_small, y_train_small = resample(X_train, y_train, n_samples=50000, random_state=42)

# ----------------------------
# Model Definition
# ----------------------------
class RTTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, num_heads=8, num_layers=4, num_classes=2):
        super(RTTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x

# ----------------------------
# Model Initialization
# ----------------------------
model = RTTransformer(
    input_dim=X_train.shape[1],
    d_model=64,
    num_heads=8,
    num_layers=4,
    num_classes=2
).to(device)

# ----------------------------
# Training Setup
# ----------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_epochs = 100

X_train_tensor = torch.tensor(X_train_small, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_small.values, dtype=torch.long)
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

# ----------------------------
# Training Loop (Updated)
# ----------------------------
train_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct_preds += (preds == batch_y).sum().item()
        total_samples += batch_y.size(0)

    avg_loss = total_loss / len(train_loader)
    train_accuracy = correct_preds / total_samples

    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)

    # Validation Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()
        val_acc = accuracy_score(y_test, test_preds)
        val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Validation Accuracy: {val_acc:.4f}")


# ----------------------------
# Evaluation Metrics and Threshold Analysis (Updated)
# ----------------------------

# Prediction on Test Set (Fixed)
# ----------------------------
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs_test = model(X_test_tensor)
    probs_test = torch.softmax(outputs_test, dim=1)[:, 1].detach().cpu().numpy()

# Compute thresholds
fpr, tpr, thresholds = roc_curve(y_test, probs_test)
ks_values = tpr - fpr
best_ks_idx = np.argmax(ks_values)
best_threshold_ks = thresholds[best_ks_idx]

precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, probs_test)
f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-6)
best_f1_idx = np.argmax(f1_scores)
best_threshold_f1 = thresholds_pr[best_f1_idx]

print(f"\n✅ Best KS Threshold: {best_threshold_ks:.4f}")
print(f"✅ Best F1 Threshold: {best_threshold_f1:.4f}")

# Apply best KS threshold
y_pred_final = (probs_test >= best_threshold_ks).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)
auc_val = roc_auc_score(y_test, probs_test)
conf_matrix = confusion_matrix(y_test, y_pred_final)


results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_final})
results_df['Positive'] = np.where(results_df['Actual'] == 1, results_df['Predicted'], 0)
results_df['Negative'] = np.where(results_df['Actual'] == 0, results_df['Predicted'], 0)
cum_pos = results_df['Positive'].cumsum() / results_df['Positive'].sum()
cum_neg = results_df['Negative'].cumsum() / results_df['Negative'].sum()
ks_statistic = np.max(np.abs(cum_pos - cum_neg))

print(f"\n--- Final Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc_val:.4f}")
print(f"KS Statistic: {ks_statistic:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# ----------------------------
# KS Curve Plot
# ----------------------------

plt.figure(figsize=(10, 6))
plt.plot(cum_pos, label='Cumulative Positive', color='blue')
plt.plot(cum_neg, label='Cumulative Negative', color='orange')
plt.axhline(y=ks_statistic, color='red', linestyle='--', label='KS Statistic')
plt.title('KS Statistic Curve')
plt.xlabel('Threshold')
plt.ylabel('Cumulative Distribution')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# ROC and Precision-Recall Curves
# ----------------------------

# ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_val:.4f})')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Scatter Plot: Training Loss vs Validation Accuracy
# ----------------------------
# Scatter Accuracy
metrics_df = pd.DataFrame({'Accuracy': train_accuracies, 'Val_Accuracy': val_accuracies})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=metrics_df, x='Accuracy', y='Val_Accuracy')
plt.title('Train vs Validation Accuracy')
plt.grid(True)
plt.show()

# ----------------------------
# Confusion Matrix
# ----------------------------
# Confusion Matrix Plot
# ----------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ----------------------------
# Confusion Matrix Plot (Advanced)
# ----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', linewidths=1.5, linecolor='black', annot_kws={"size": 14})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix (Enhanced)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.show()


# ----------------------------
# Accuracy and Loss Plots (اگر train_losses, val_accuracies ذخیره شده باشند)
# ----------------------------
if 'train_losses' in globals() and 'val_accuracies' in globals():
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# Train Loss per Epoch
plt.figure(figsize=(8,5))
plt.plot(train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Train Accuracy per Epoch
plt.figure(figsize=(8,5))
plt.plot(train_accuracies, marker='o', color='green', label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Precision-Recall Curve
# ----------------------------
plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.legend()
plt.show()


final_targets = y_test
final_probs = probs_test
# ----------------------------
# Gain Chart
# ----------------------------
results_df = pd.DataFrame({'y_true': final_targets, 'y_score': final_probs}).sort_values('y_score', ascending=False).reset_index(drop=True)
results_df['Cumulative Positive Rate'] = results_df['y_true'].cumsum() / results_df['y_true'].sum()
results_df['Cumulative Total Rate'] = (np.arange(len(results_df)) + 1) / len(results_df)

plt.figure(figsize=(8,6))
plt.plot(results_df['Cumulative Total Rate'], results_df['Cumulative Positive Rate'], label='Model Gain')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Baseline')
plt.xlabel('Proportion of Sample')
plt.ylabel('Proportion of Positive Class Captured')
plt.title('Gain Chart')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Lift Chart
# ----------------------------
results_df['Lift'] = results_df['Cumulative Positive Rate'] / results_df['Cumulative Total Rate']

plt.figure(figsize=(8,6))
plt.plot(results_df['Cumulative Total Rate'], results_df['Lift'], label='Lift')
plt.axhline(y=1, linestyle='--', color='gray', label='Baseline Lift')
plt.xlabel('Proportion of Sample')
plt.ylabel('Lift')
plt.title('Lift Chart')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# SHAP Analysis using SimpleModel
# ----------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(X_train.shape[1], 1)

    def forward(self, x):
        return self.layer(x)

shap_model = SimpleModel().to(device)

# Prediction function for SHAP
def model_predict(x):
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    if x_tensor.ndim == 1:
        x_tensor = x_tensor.unsqueeze(0)
    with torch.no_grad():
        return shap_model(x_tensor).cpu().numpy()

# SHAP Explainer
explainer = shap.Explainer(model_predict, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, feature_names=features.columns.tolist())

# ----------------------------
# Save the results to a new CSV file
data.to_csv('customer_credit_offers_RT_Transformer.csv', index=False)
files.download('customer_credit_offers_RT_Transformer.csv')


