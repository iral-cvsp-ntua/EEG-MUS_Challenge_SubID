import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        Q = self.query(x)  # [batch_size, seq_len, feature_dim]
        K = self.key(x)    # [batch_size, seq_len, feature_dim]
        V = self.value(x)  # [batch_size, seq_len, feature_dim]
        # Compute attention scores
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attention = F.softmax(attention, dim=-1)
        # Apply attention to values
        x = torch.matmul(attention, V)  # [batch_size, seq_len, feature_dim]
        return x, attention

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        args_defaults = dict(
            in_channels=1,
            num_classes=40,
        )
        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        # Create MobileNet encoder
        self.encoder = timm.create_model('tf_mobilenetv3_large_minimal_100',
                                       pretrained=True,
                                       features_only=True)
        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        feature_dim = self.encoder.feature_info[-1]['num_chs']
        # Add temporal attention layer
        self.temporal_attention = TemporalAttention(feature_dim)
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, self.num_classes),
        )
    def forward(self, x):
        # Input shape: [batch_size, seq_len, height, width]
        # Need: [batch_size * seq_len, channels, height, width]
        #print('in',x.shape)
        #x = x.unsqueeze(1)
        batch_size, seq_len, height, width = x.shape
        # Unsqueeze to add channel dimension
        x = x.unsqueeze(2)  # [batch_size, seq_len, 1, height, width]
        # Reshape for MobileNet processing
        x = x.view(batch_size * seq_len, 1, height, width)
        # Repeat channels for MobileNet
        x = x.repeat(1, 3, 1, 1)
        #print('pre-enc',x.shape)
        # Extract features
        features = self.encoder(x)[-1]  # [batch_size * seq_len, feature_dim, h’, w’]
        # Global average pooling
        features = F.avg_pool2d(features, features.size()[2:]).view(batch_size * seq_len, -1)
        # Reshape back to include sequence dimension
        features = features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, feature_dim]
        # Apply temporal attention
        attended_features, attention_weights = self.temporal_attention(features)
        # Use the final timestep features for classification
        final_features = attended_features[:, -1, :]  # [batch_size, feature_dim]
        # Classification
        out = self.classifier(final_features)
        return out#, attention_weights
