import torch
import torch.nn as nn
import torch.nn.functional as F

class AcousticModel(nn.Module):
    """
    The Convolutional Stack used for feature extraction.
    Standard Config: 3 layers, 48 filters, 3x3 kernels, (1,2) pooling.
    """
    def __init__(self, input_channels, model_complexity=48, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, model_complexity, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(model_complexity)
        self.pool1 = nn.MaxPool2d((1, 2))

        self.conv2 = nn.Conv2d(model_complexity, model_complexity, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(model_complexity)
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(model_complexity, model_complexity, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(model_complexity)
        self.pool3 = nn.MaxPool2d((1, 2))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 1, T, F)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Output shape: (B, C, T, F_pooled)
        return x

class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features=229, output_features=88, model_complexity=48, lstm_units=128):
        super().__init__()
        
        # Calculate the flattened size after 3 poolings of (1, 2)
        # 229 -> 114 -> 57 -> 28
        # This math assumes floor division.
        # It's safer to compute dynamically, but for standard Mel specs:
        f_pooled = input_features
        for _ in range(3):
            f_pooled = f_pooled // 2
        
        self.cnn_output_size = model_complexity * f_pooled
        self.lstm_units = lstm_units
        
        # --- 1. Onset Branch ---
        self.onset_stack = AcousticModel(1, model_complexity)
        self.onset_rnn = nn.LSTM(
            input_size=self.cnn_output_size, 
            hidden_size=lstm_units, 
            batch_first=True, 
            bidirectional=True
        )
        self.onset_fc = nn.Linear(lstm_units * 2, output_features)

        # --- 2. Frame Branch ---
        self.frame_stack = AcousticModel(1, model_complexity)
        
        # Projection layer to match sizes before RNN (found in GitHub repo)
        self.frame_proj = nn.Linear(self.cnn_output_size, output_features)
        
        # Frame RNN Input = [Projected Frame Features] + [Onset Probabilities]
        self.frame_rnn = nn.LSTM(
            input_size=output_features + output_features, 
            hidden_size=lstm_units, 
            batch_first=True, 
            bidirectional=True
        )
        self.frame_fc = nn.Linear(lstm_units * 2, output_features)

        # --- 3. Velocity Branch ---
        # Shares the Onset RNN output
        self.velocity_fc = nn.Linear(lstm_units * 2, output_features)

    def forward(self, x):
        # x: (Batch, Time, Freq)
        # Add channel dim: (B, 1, T, F)
        x = x.unsqueeze(1)
        B, C, T, F = x.shape

        # =========================
        # ONSET BRANCH
        # =========================
        # 1. Acoustic Model
        onset_features = self.onset_stack(x) # (B, 48, T, F_pooled)
        
        # 2. Reshape for RNN: (B, T, 48 * F_pooled)
        onset_features = onset_features.permute(0, 2, 1, 3).contiguous()
        onset_features = onset_features.view(B, T, -1)
        
        # 3. BiLSTM
        onset_rnn_out, _ = self.onset_rnn(onset_features)
        
        # 4. Prediction
        onset_logits = self.onset_fc(onset_rnn_out)
        onset_probs = torch.sigmoid(onset_logits)

        # =========================
        # FRAME BRANCH
        # =========================
        # 1. Acoustic Model
        frame_features = self.frame_stack(x)
        frame_features = frame_features.permute(0, 2, 1, 3).contiguous()
        frame_features = frame_features.view(B, T, -1)
        
        # 2. Projection & Activation (Sigmoid is typically used here in repo)
        frame_features = torch.sigmoid(self.frame_proj(frame_features))
        
        # 3. Concatenate with Onset Probs
        # STOP GRADIENT: We do not want Frame loss to update Onset weights
        # The paper emphasizes this "detach" logic.
        frame_input = torch.cat([frame_features, onset_probs.detach()], dim=2)
        
        # 4. BiLSTM
        frame_rnn_out, _ = self.frame_rnn(frame_input)
        
        # 5. Prediction
        frame_logits = self.frame_fc(frame_rnn_out)

        # =========================
        # VELOCITY BRANCH
        # =========================
        # Typically just a linear layer off the Onset RNN
        velocity_logits = self.velocity_fc(onset_rnn_out)

        return {
            "onset": onset_logits,
            "frame": frame_logits,
            "velocity": velocity_logits,
            "onset_probs": onset_probs
        }