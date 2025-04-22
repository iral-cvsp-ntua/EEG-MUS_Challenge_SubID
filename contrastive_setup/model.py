import torch
import torch.nn.functional as F
from torch import nn
import timm
from models.eegchannelnet import *

class ContrastiveMobileNet(nn.Module):
    def __init__(self):# in_channels):
        super(ContrastiveMobileNet, self).__init__()

        output_dim = 256
        #self.in_channels = in_channels
        # Encoder: Pretrained MobileNetV3
        self.encoder = timm.create_model('tf_mobilenetv3_large_minimal_100', 
                                    pretrained=False,
                                    features_only=True)  # Keeps only the feature extraction layers
        
                # Dense layers for  EEG and eeg2 embeddings 
        self.eeg_dense = nn.Sequential(
            nn.Linear(960, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.eeg2_dense = nn.Sequential(
            nn.Linear(960, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

        
    
    def _forward(self,x):    
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add a channel dimension if input is 3D (e.g., EEG data)
        
        x = x.repeat(1, 3, 1, 1)  # Repeat across 3 channels if using grayscale input (for RGB models)
        #print(f'Input after repeating channels: {x.shape}')

        # Encoder forward pass
        features_prior = self.encoder(x)
        #print(features_prior[-1].shape)
        features = features_prior[-1]  # Use the last feature map from the encoder
        features = F.avg_pool2d(features, features.size()[2:]).view(features.size(0), -1)
        return features
    
    def forward(self, eeg_data, eeg2_data):
        if len(eeg_data.shape) == 3:
            eeg_data = eeg_data.unsqueeze(1)
        #print('a',eeg_data.shape)
        # Process eeg2 embeddings through the dense layers
        if len(eeg2_data.shape) == 3:
            eeg2_data = eeg2_data.unsqueeze(1)

        #print((eeg_data - eeg2_data).shape

        #print(eeg_data.shape)
        outa = self._forward(eeg_data)

        #print(outa.shape)
        
        #outa = outa.contiguous().view(eeg_data.size(0), -1)
        
        eeg_features = self.eeg_dense(outa)  # Apply the dense layers

        #print('b',eeg2_data.shape)
        # Process EEG data through the EEG network
        outb = self._forward(eeg2_data)

        outb = outb.contiguous().view(eeg2_data.size(0), -1)

        eeg2_features = self.eeg_dense(outb) 

        # Normalize the features for cosine similarity
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        eeg2_features = F.normalize(eeg2_features, p=2, dim=-1)

        return eeg_features, eeg2_features

    def contrastive_loss(self, eeg_features, eeg2_features):
        # Compute cosine similarity between all pairs in the batch
        similarity_matrix = (torch.matmul(eeg_features, eeg2_features.T))
        normalized_simaccs = torch.argmax(F.softmax(similarity_matrix),axis=0)
        #print(F.softmax(similarity_matrix))

        batch_size = eeg_features.size(0)
        labels = torch.arange(batch_size).to(eeg_features.device)

        #print(normalized_simaccs,labels, torch.sum(normalized_simaccs==labels))
        loss_eeg = F.cross_entropy(similarity_matrix, labels)
        loss_eeg2 = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_eeg+loss_eeg2) / 2.0, torch.sum(normalized_simaccs == labels)

class ContrastiveModel(nn.Module):
    def __init__(self, output_dim=256,num_channels=32):

        super(ContrastiveModel, self).__init__()
        
        self.num_channels=num_channels
        
        self.conv1 = nn.Conv2d(1, 16, (1, self.num_channels), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        self.encoding_size = 640 #self.encoder(torch.zeros(1, 1, 32, segment_size_seconds*128)).contiguous().view(-1).size()[0]

        # Dense layers for  EEG and eeg2 embeddings 
        self.eeg_dense = nn.Sequential(
            nn.Linear(self.encoding_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.eeg2_dense = nn.Sequential(
            nn.Linear(self.encoding_size, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )


    def _forward(self,x):

 # Process EEG data through the EEG network
        #x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        #print(x.shape)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        return x
    
    def forward(self, eeg_data, eeg2_data):
        if len(eeg_data.shape) == 3:
            eeg_data = eeg_data.unsqueeze(1)
        #print('a',eeg_data.shape)
        # Process eeg2 embeddings through the dense layers
        if len(eeg2_data.shape) == 3:
            eeg2_data = eeg2_data.unsqueeze(1)

        #print((eeg_data - eeg2_data).shape

        #print(eeg_data.shape)
        outa = self._forward(eeg_data)

        #print(outa.shape)
        
        outa = outa.contiguous().view(eeg_data.size(0), -1)
        
        eeg_features = self.eeg_dense(outa)  # Apply the dense layers

        #print('b',eeg2_data.shape)
        # Process EEG data through the EEG network
        outb = self._forward(eeg2_data)

        outb = outb.contiguous().view(eeg2_data.size(0), -1)

        eeg2_features = self.eeg_dense(outb) 

        # Normalize the features for cosine similarity
        eeg_features = F.normalize(eeg_features, p=2, dim=-1)
        eeg2_features = F.normalize(eeg2_features, p=2, dim=-1)

        return eeg_features, eeg2_features

    def contrastive_loss(self, eeg_features, eeg2_features):
        # Compute cosine similarity between all pairs in the batch
        similarity_matrix = (torch.matmul(eeg_features, eeg2_features.T))
        normalized_simaccs = torch.argmax(F.softmax(similarity_matrix),axis=0)
        #print(F.softmax(similarity_matrix))

        batch_size = eeg_features.size(0)
        labels = torch.arange(batch_size).to(eeg_features.device)

        #print(normalized_simaccs,labels, torch.sum(normalized_simaccs==labels))
        loss_eeg = F.cross_entropy(similarity_matrix, labels)
        loss_eeg2 = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_eeg+loss_eeg2) / 2.0, torch.sum(normalized_simaccs == labels)
