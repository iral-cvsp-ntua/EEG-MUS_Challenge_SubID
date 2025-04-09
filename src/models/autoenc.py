import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
import timm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()

        args_defaults =  dict(in_channels=32, d_model=128, n_heads=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1, max_len=5000,n_classes=26)

        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        #print(in_channels)
        # Input embedding layers (from in_channels to d_model size)
        self.encoder_input_layer = nn.Linear(32,self.d_model)
        self.decoder_input_layer = nn.Linear(32, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, dropout=self.dropout, max_len=self.max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.dim_feedforward, dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)

        # Transformer Decoder
        #decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        #self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output linear layer to map back to input channel size
        self.classifier = nn.Linear(self.d_model,self.n_classes)

        # Masking (for autoregressive behavior, you may not need this for autoencoder)
        self.src_mask = None
        self.tgt_mask = None

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        
        batch_size, in_channels, seq_len = x.shape

        #print(x.shape)
        # [batch_size, in_channels, seq_len] -> [seq_len, batch_size, in_channels]
        #print(f'TRANSFORMER {src.shape}')
        x = x.permute(2, 0, 1)  # Transformer expects [seq_len, batch_size, in_channels]
        #print(f'after permutation {src.shape}')

        #print('post permute',x.shape)
        # Input embedding and positional encoding for encoder
        encoder_input = self.encoder_input_layer(x)
        #print(f'after encoder input layer {encoder_input.shape}')
        encoder_input = self.positional_encoding(encoder_input)
        #print(f'after positinal embedding {encoder_input.shape}')

        self.src_mask = None
        # Generate masks for sequence data if necessary
        if self.src_mask is None or self.src_mask.size(0) != seq_len:
            self.src_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)


        # Encoder forward pass
        memory = self.encoder(encoder_input, mask=self.src_mask)
        memory = memory.permute(1, 2, 0)
            #print(f'AFTER ENCODER WITH RESHAPE {eeg_features.shape}')
        memory = torch.mean(memory, dim=-1)
        #memory = memory[-1,:,:]
        #print(f'after encoder input {memory.shape}')
        
        output = self.classifier(memory)
        # Input embedding and positional encoding for decoder
        #decoder_input = self.decoder_input_layer(src)
        #decoder_input = self.positional_encoding(decoder_input)

        # Apply the decoder
        #output = self.decoder(tgt=decoder_input, memory=memory, tgt_mask=self.tgt_mask, memory_mask=self.src_mask)

        # Final output layer to reconstruct the input
        #output = self.output_layer(output)

        # [seq_len, batch_size, in_channels] -> [batch_size, in_channels, seq_len]
        #output = output.permute(1, 2, 0)

        #
        # print(output.shape)
        return output

    def loss(self, recon_x, x):
        """ MSE Loss """
        print(recon_x.shape,x.shape)
        return F.mse_loss(recon_x, x, reduction='mean')

