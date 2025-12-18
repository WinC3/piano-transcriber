import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_parser import PianoRollDataset
from nn_models import OnsetsAndFrames
import os
import sys

def load_checkpoint(model, optimizer, checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    start_epoch = ckpt.get('epoch', 0)
    loss_val = ckpt.get('loss', None)

    print(f"Checkpoint loaded (epoch {start_epoch})")
    return start_epoch

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
TRAIN_DATA_DIR = "processed_maestro/train"
VAL_DATA_DIR = "processed_maestro/validation"
CHECKPOINT_DIR = "checkpoints"

class Metrics:
    """
    Simple accumulator for Precision, Recall, F1, and Accuracy.
    Calculates frame-wise metrics (good proxy for training progress).
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.count = 0

    def update(self, logits, targets, threshold=0.5):
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        targets = targets.float()

        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()
        self.tn += ((1 - preds) * (1 - targets)).sum().item()
        self.count += 1

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-8)
        return precision, recall, f1, accuracy

def get_user_input(current_lr):
    print("\n--- Training Control ---")
    while True:
        try:
            epochs_str = input(f"Enter number of epochs to train (0 to stop): ")
            epochs = int(epochs_str)
            if epochs < 0:
                print("Please enter a positive number.")
                continue
            
            if epochs == 0:
                return 0, current_lr

            lr_str = input(f"Enter Learning Rate (current: {current_lr}) [Press Enter to keep]: ")
            if lr_str.strip() == "":
                new_lr = current_lr
            else:
                new_lr = float(lr_str)
            
            return epochs, new_lr
        except ValueError:
            print("Invalid input. Please enter numbers.")

def train():
    # 1. Setup Data
    if not os.path.exists(TRAIN_DATA_DIR):
        print(f"Error: Directory {TRAIN_DATA_DIR} not found. Run preprocessing first.")
        return

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dataset = PianoRollDataset(TRAIN_DATA_DIR, is_validation=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = PianoRollDataset(VAL_DATA_DIR, is_validation=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Init Model
    model = OnsetsAndFrames().to(DEVICE)
    
    # Default starting LR
    current_lr = 0.0006
    optimizer = optim.Adam(model.parameters(), lr=current_lr)

    # ---- NEW CODE FOR RESUMING ----
    resume_choice = input("Resume from checkpoint? (y/n): ").strip().lower()

    start_epoch = 0
    if resume_choice == "y":
        ckpt_path = input("Enter checkpoint path (e.g., checkpoints/model_epoch_12.pth): ").strip()
        if os.path.exists(ckpt_path):
            start_epoch = load_checkpoint(model, optimizer, ckpt_path, DEVICE)
            # also restore LR if changed in optimizer
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            print("Checkpoint path not found. Starting from scratch.")
    # --------------------------------
    total_epochs_trained = start_epoch
    
    # Losses
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss(reduction='none') 

    print(f"Model initialized on {DEVICE}")

    # 3. Interactive Loop
    while True:
        epochs_to_run, current_lr = get_user_input(current_lr)
        
        if epochs_to_run == 0:
            print("Stopping training.")
            break

        # Update Optimizer LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"Training for {epochs_to_run} epochs with LR={current_lr}...")

        for epoch in range(epochs_to_run):
            model.train()
            
            # Metrics for this epoch
            train_onset_metrics = Metrics()
            train_frame_metrics = Metrics()
            total_train_loss = 0
            
            # --- TRAINING PHASE ---
            print(f"\nEpoch {total_epochs_trained + 1} (Training)...")
            for batch_idx, batch in enumerate(train_loader):
                audio = batch['audio'].to(DEVICE)
                onset_label = batch['onset'].to(DEVICE)
                frame_label = batch['frame'].to(DEVICE)
                velocity_label = batch['velocity'].to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(audio)
                
                # Calculate Losses
                loss_onset = bce_loss(outputs['onset'], onset_label)
                loss_frame = bce_loss(outputs['frame'], frame_label)
                
                vel_loss_raw = mse_loss(outputs['velocity'], velocity_label)
                onset_mask = (onset_label == 1).float()
                loss_velocity = (vel_loss_raw * onset_mask).sum() / (onset_mask.sum() + 1e-6)
                
                loss = loss_onset + loss_frame + loss_velocity
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                
                # Update Metrics
                train_onset_metrics.update(outputs['onset'], onset_label)
                train_frame_metrics.update(outputs['frame'], frame_label)
                total_train_loss += loss.item()

                if batch_idx % 1 == 0:
                    # Quick print of loss to show life
                    print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.8f}", end='\r')

            # --- VALIDATION PHASE ---
            model.eval()
            val_loss = 0
            val_onset_metrics = Metrics()
            val_frame_metrics = Metrics()

            with torch.no_grad():
                for batch in val_loader:
                    audio = batch['audio'].to(DEVICE)
                    onset_label = batch['onset'].to(DEVICE)
                    frame_label = batch['frame'].to(DEVICE)
                    velocity_label = batch['velocity'].to(DEVICE)
                    
                    outputs = model(audio)
                    
                    l_o = bce_loss(outputs['onset'], onset_label)
                    l_f = bce_loss(outputs['frame'], frame_label)
                    val_loss += (l_o + l_f).item()

                    val_onset_metrics.update(outputs['onset'], onset_label)
                    val_frame_metrics.update(outputs['frame'], frame_label)

            # --- EPOCH SUMMARY ---
            # Compute final metrics
            tr_o_p, tr_o_r, tr_o_f1, tr_o_acc = train_onset_metrics.compute()
            tr_f_p, tr_f_r, tr_f_f1, tr_f_acc = train_frame_metrics.compute()
            
            val_o_p, val_o_r, val_o_f1, val_o_acc = val_onset_metrics.compute()
            val_f_p, val_f_r, val_f_f1, val_f_acc = val_frame_metrics.compute()

            print(f"\nCompleted Epoch {total_epochs_trained + 1}")
            print("-" * 60)
            print(f"Train Loss: {total_train_loss/len(train_loader):.8f} | Val Loss: {val_loss/len(val_loader):.8f}")
            print(f"ONSET ACC - Train: {tr_o_acc:.5f} | Val: {val_o_acc:.5f}")
            print(f"FRAME ACC - Train: {tr_f_acc:.5f} | Val: {val_f_acc:.5f}")
            print(f"ONSET F1  - Train: {tr_o_f1:.8f} (P:{tr_o_p:.8f} R:{tr_o_r:.8f}) | Val: {val_o_f1:.8f} (P:{val_o_p:.8f} R:{val_o_r:.8f})")
            print(f"FRAME F1  - Train: {tr_f_f1:.8f} (P:{tr_f_p:.8f} R:{tr_f_r:.8f}) | Val: {val_f_f1:.8f} (P:{val_f_p:.8f} R:{val_f_r:.8f})")
            print("-" * 60)

            total_epochs_trained += 1

            # Save Checkpoint
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{total_epochs_trained}.pth")
            torch.save({
                'epoch': total_epochs_trained,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    train()