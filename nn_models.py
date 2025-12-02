import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        # Simulating the acoustic model structure
        # (Batch, 1, T, F) -> Conv -> ...
        self.conv1 = nn.Conv2d(1, output_features // 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(output_features // 16)
        
        self.conv2 = nn.Conv2d(output_features // 16, output_features // 8, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(output_features // 8)
        
        self.pool = nn.MaxPool2d((1, 2)) # Pool frequency only
        
        self.conv3 = nn.Conv2d(output_features // 8, output_features, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(output_features)
        
        self.fc = nn.Linear(input_features // 2 * output_features, output_features) # rough calculation for flattened dim

    def forward(self, x):
        # x: (B, 1, T, F_in)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Prepare for RNN: (B, Channels, T, F_pooled)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 88) # Project to piano keys

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.fc(x) # Returns raw logits

class OnsetsAndFrames(nn.Module):
    def __init__(self, input_bins=229, model_complexity=48):
        super().__init__()
        
        # --- Acoustic Models ---
        # We can share the conv stack or have separate ones. 
        # The paper often uses separate stacks but similar architecture.
        # Calculating the flattened size after one pool(1, 2) roughly: input_bins / 2
        conv_out_features = model_complexity * (input_bins // 2)
        
        # Onset specific layers
        self.onset_conv = nn.Sequential(
            nn.Conv2d(1, model_complexity, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(model_complexity),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(model_complexity, model_complexity, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(model_complexity),
            nn.ReLU(),
            nn.MaxPool2d((1,2))
        )
        # Calculate RNN input size based on Mel bins=229 and 2 pools
        rnn_input_size = model_complexity * (input_bins // 4) 
        
        self.onset_rnn = nn.LSTM(rnn_input_size, 256, batch_first=True, bidirectional=True)
        self.onset_fc = nn.Linear(512, 88)

        # Frame specific layers
        self.frame_conv = nn.Sequential(
            nn.Conv2d(1, model_complexity, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(model_complexity),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(model_complexity, model_complexity, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(model_complexity),
            nn.ReLU(),
            nn.MaxPool2d((1,2))
        )
        
        # Frame RNN input = Frame Conv features + Onset Predictions (88)
        self.frame_rnn = nn.LSTM(rnn_input_size + 88, 256, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(512, 88)
        
        # Velocity specific layers (shared onset conv usually)
        self.velocity_fc = nn.Linear(512, 88)

    def forward(self, x):
        # x: (B, T, F) -> needs (B, 1, T, F) for Conv2d
        x = x.unsqueeze(1)
        B, C, T, F = x.shape

        # --- Onset Branch ---
        o_features = self.onset_conv(x) # (B, C, T, F/4)
        # Flatten for RNN
        o_features = o_features.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        
        o_rnn_out, _ = self.onset_rnn(o_features)
        onset_logits = self.onset_fc(o_rnn_out)
        onset_probs = torch.sigmoid(onset_logits) # Used for Frame input

        # --- Frame Branch ---
        f_features = self.frame_conv(x)
        f_features = f_features.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        
        # Concatenate acoustic features with Onset Probabilities (detached or not? Paper doesn't detach)
        # We assume alignment is perfect here.
        f_input = torch.cat([f_features, onset_probs.detach()], dim=2) 
        # Note: .detach() is a hyperparameter decision. 
        # If you want gradients to flow from Frame loss -> Onset weights, remove detach.
        # Standard O&F usually does NOT detach. 
        # f_input = torch.cat([f_features, onset_probs], dim=2)
        
        f_rnn_out, _ = self.frame_rnn(f_input)
        frame_logits = self.frame_fc(f_rnn_out)
        
        # --- Velocity Branch ---
        # Uses Onset RNN features
        velocity_logits = self.velocity_fc(o_rnn_out)

        return {
            "onset": onset_logits,       # Return logits for BCEWithLogitsLoss
            "frame": frame_logits,       # Return logits for BCEWithLogitsLoss
            "velocity": velocity_logits, # Return raw linear output
            "onset_probs": onset_probs   # Return probs for inference/connection
        }