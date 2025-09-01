import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch

import data_parser as dp


class PitchDetector(nn.Module):
    def __init__(self, n_layers, n_input_bins=88, n_notes=88):
        super().__init__()

        n_intermediate_layers = n_layers - 2
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_input_bins, 256))
        for i in range(n_intermediate_layers):
            self.layers.append(nn.Linear(256, 256))
        self.layers.append(nn.Linear(256, n_notes))

        '''# encoder layers
        step_enc = (num_question - k) // enc_layers
        self.encoder = nn.ModuleList()
        prev = num_question
        for i in range(enc_layers):
            next_size = num_question - (i + 1) * step_enc
            if i == enc_layers - 1:
                next_size = k
            self.encoder.append(nn.Linear(prev, next_size))
            prev = next_size'''

    def get_weight_norm(self):
        return sum(torch.norm(m.weight, 2) ** 2
                   for m in self.modules()
                   if isinstance(m, nn.Linear))

    def forward(self, inputs):
        
        out = inputs
        for layer in self.layers[:-1]:
            out = torch.relu(layer(out))
        out = torch.sigmoid(self.layers[-1](out))

        return out


def train(model, lr, lamb, train_data, train_labels, valid_data, valid_labels, 
          num_epoch, batch_size=128):

    # Create datasets and loaders
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_data, valid_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss += (lamb / 2) * model.get_weight_norm()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)  # rescale by batch for total loss

        avg_train_loss = train_loss / len(train_loader.dataset)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                preds = (outputs >= 0.5).float()
                correct += (preds == targets).sum().item()
                total += targets.numel()

        val_acc = correct / total
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epoch}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_accuracies


def ensure_tensor(data, dtype=torch.float32):
    """Convert data to tensor if it isn't already"""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(dtype)
    else:
        return torch.tensor(data, dtype=dtype)


def evaluate(model, data, labels, batch_size=None):
    model.eval()
    
    if batch_size is None:
        batch_size = len(data)
    
    data = data[:batch_size]
    labels = labels[:batch_size]
    
    with torch.no_grad():
        outputs = model(data)
        predictions = (outputs >= 0.5).float()
        correct = (predictions == labels).sum().item()
        total = labels.numel()
    
    return correct / total


def ask_continue():
    resp = input("Enter number of epochs to continue training, or 'q' to quit: ")
    if resp.lower() == 'q':
        return None
    try:
        return int(resp)
    except ValueError:
        print("Invalid input. Please enter a number or 'q'.")
        return ask_continue()


def main():
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = dp.load_dataset_from_file(n_samples=100000, shuffle=True)
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    print(f"Valid data shape: {valid_data.shape}, Valid labels shape: {valid_labels.shape}")
    print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

    # Convert to tensors
    train_data = ensure_tensor(train_data)
    train_labels = ensure_tensor(train_labels)
    valid_data = ensure_tensor(valid_data)
    valid_labels = ensure_tensor(valid_labels)
    test_data = ensure_tensor(test_data)
    test_labels = ensure_tensor(test_labels)

    # hyperparameters
    lr = 0.03
    #num_epoch = 0
    lamb = 0
    n_layers = 4

    train_accs, val_accs, test_accs = [], [], []

    # nn model
    model = PitchDetector(n_layers=n_layers, n_input_bins=train_data.shape[1], n_notes=88)

    cur_epoch = 0
    while num_epoch := ask_continue():
        if num_epoch is None:
            print("Exiting training loop.")
            break
        for epoch in range(num_epoch):
            train_losses, val_accuracies = train(model, lr, lamb, train_data, train_labels, valid_data, valid_labels, num_epoch=1)

            train_acc = evaluate(model, train_data, train_labels)
            val_acc = evaluate(model, valid_data, valid_labels)
            test_acc = evaluate(model, test_data, test_labels)

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)

            print(f"Final Model Epoch {cur_epoch} \t Train Acc: {train_acc:.4f} \t Valid Acc: {val_acc:.4f} \t Test Acc: {test_acc:.4f}")
            cur_epoch += 1


    # plot training loss and validation accuracy
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # final model plots
    plt.plot(range(num_epoch), train_accs, label='Training Accuracy', color='blue')
    plt.plot(range(num_epoch), val_accs, label='Validation Accuracy', color='green')
    plt.axhline(y=test_acc, linestyle='--', color='red', label='Final Model Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Final Model Accuracies over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"lr: {lr}, epochs: {num_epoch}, layers: {n_layers}.png")
    #plt.show()


if __name__ == "__main__":
    main()
