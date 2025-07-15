#--------------------------Waveform Database Libraries----------------------------------------------------------

import pandas as pd
import scipy.io
import os
import wfdb


#--------------------------Read and Combine Funtions----------------------------------------------------------

def read_mat_files(folder_path, label):
    data_list = [] 
    label_list = []

    #Loads all the mat files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mat"):
            file_path = os.path.join(folder_path, filename)
            mat_data = scipy.io.loadmat(file_path)
            
            data_list.append(mat_data)
            label_list.append(label)

    return data_list, label_list

def combine_data(folder_paths):
    all_data = []
    all_labels = []
    signals_array = []

    #
    for label, folder_path in enumerate(folder_paths):
        data_list, labels = read_mat_files(folder_path, label)
        all_data.extend(data_list)
        all_labels.extend(labels)

        # Process each .mat file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".mat"):
                file_path = os.path.join(folder_path, filename)
                record = wfdb.rdrecord(file_path.replace(".mat", ""))
                
                for idx, signal in enumerate(record.p_signal.T):
                    signals_array.append(signal)

    # Convert to DataFrame
    dataFrame = pd.DataFrame(signals_array)
    labelSeries = pd.Series(all_labels, name='label')

    return dataFrame, labelSeries


#---------------------------------Defining Database Structure---------------------------------------------------

# Determines path to each class folder
folder_paths = [
    '/Users/atorN/Dropbox/ECGs_training2017/Class_A',
    '/Users/atorN/Dropbox/ECGs_training2017/Class_N'  #Removed Folder O Since we want binary classification
]

# Combine data
dataFrame, labelSeries = combine_data(folder_paths)

# Verify that DataFrame and Series are created properly
print(f"DataFrame shape: {dataFrame.shape}")
print(f"Label Series shape: {labelSeries.shape}")

# Combine DataFrame and Series
dataFrame = pd.concat([labelSeries, dataFrame], axis=1)

# Randomize the rows in the DataFrame
dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)

# Verify the combined DataFrame
print(f"Combined and randomized DataFrame shape: {dataFrame.shape}")

# Save to CSV
try:
    dataFrame.to_csv('combined_ECG_Data.csv', index=False)
    print("CSV file created successfully.")
except Exception as e:
    print(f"Error saving CSV file: {e}")

#---------------------------CNN Libraries---------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset, random_split
import time

#---------------------------------Designating Training and Testing Sets---------------------------------------------------

# --- Train and Test split manually (test with patient 233 and 234 ECG windows) ---
# --- Dataset: Previously sectioned ECG recordings into 2-second (360 Hz) windows ---

train = dataFrame.iloc[0:2000] 
test = dataFrame.iloc[2001:]

sub_timewindow = 1000

# Print the shape of train to understand its dimensions
print("Shape of train DataFrame:", train.shape)

X_train = train.iloc[:, 0:sub_timewindow].values  # voltages, train
X_test = test.iloc[:, 0:sub_timewindow].values    # voltages, test
Y_train = train['label'].values      # results, train
Y_test = test['label'].values        # results, test

print('Train Shape - voltages, label:')
print(X_train.shape, Y_train.shape)
print('Test Shape - voltages, label:')
print(X_test.shape, Y_test.shape)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# --- Create training and validation sets ---
# Split the training set into training and validation
val_split = 0.2
train_size = int((1 - val_split) * len(X_train_tensor))
val_size = len(X_train_tensor) - train_size
train_dataset, val_dataset = random_split(TensorDataset(X_train_tensor, Y_train_tensor), [train_size, val_size])

# DataLoaders for training, validation, and test sets
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=batch_size, shuffle=False)

# Print the sample sizes
print(f"Training set size: {len(train_loader.dataset)}")
print(f"Validation set size: {len(val_loader.dataset)}")
print(f"Testing set size: {len(test_loader.dataset)}")

#---------------------------------Defining Model---------------------------------------------------

# Define the model architecture in PyTorch
class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10)
        self.pool = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Equivalent to GlobalAveragePooling1D
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Reshape to (batch_size, channels, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.global_avg_pool(x).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x

# Model instance
model = ECGModel()
print(model)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Log hyperparameters
print(f"Neural network hyperparameters:\n- Number of layers: 3 convolutional layers\n- Learning rate: 0.001\n- Batch size: {batch_size}")

#---------------------------------Training and Validation Loop---------------------------------------------------

# Epochs was 100
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses, val_losses = [], []

for epoch in range(epochs):
    # Training Phase
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation Phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


# ---------------------------------- Plot Training and Validation Loss ----------------------------------

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs', fontsize = 14)
plt.ylabel('Loss', fontsize = 14)
plt.axvline(x=40, color='r', linestyle='--', alpha=0.5)

plt.annotate('Underfitting',
             fontsize=16,
             color='red',
             xy=(5, 0.3),        # Point to annotate
             xytext=(10, 0.4),   # Text location
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.annotate('Overfitting',
             fontsize=16,
             color='red',
             xy=(85, 0.15),      
             xytext=(75, 0.3), 
             arrowprops=dict(facecolor='red', shrink=0.05))
             
plt.title('Training and Validation Loss', fontsize = 20)
plt.legend(fontsize = 14)
plt.show()

# ------------------------------------- Evaluation on Test Set -------------------------------------

model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        y_pred.extend(outputs.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

y_pred = np.array(y_pred).flatten()
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
auroc = roc_auc_score(y_true, y_pred)
auprc = average_precision_score(y_true, y_pred)

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_binary))

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUROC: {auroc:.4f}')
print(f'AUPRC: {auprc:.4f}')

# ---------------------------------- Plot AUROC and AUPRC curves ----------------------------------

# Plot AUROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f'AUROC: {auroc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('AUROC Curve', fontsize=20)
plt.legend(fontsize=14)
plt.show()

# Plot AUPRC curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)
plt.figure(figsize=(10, 5))
plt.plot(recall, precision, label=f'AUPRC: {auprc:.4f}')
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('AUPRC Curve', fontsize=20)
plt.legend(fontsize=14)
plt.show()