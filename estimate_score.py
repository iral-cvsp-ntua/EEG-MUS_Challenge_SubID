import numpy as np
import torch
from sklearn import metrics
import sys

#if len(sys.argv) > 2:
run_names = sys.argv[1:-1]
splitnum = sys.argv[-1]

epoch_outputs = torch.nn.Softmax()(torch.Tensor(np.load('outputs_'+sys.argv[1]+'_'+sys.argv[-1]+'.npy')))
if len(sys.argv) > 3:
    for i in range(2,len(sys.argv)-1):
        epoch_outputs += torch.nn.Softmax()(torch.Tensor(np.load('outputs_'+sys.argv[i]+'_'+sys.argv[-1]+'.npy')))
        
epoch_labels = np.load('labels.npy')
print(epoch_outputs.shape)
if epoch_outputs.dim() > 1:
    epoch_outputs = epoch_outputs.argmax(dim=1)

#print(len(epoch_labels),len(epoch_outputs))
# Accuracy
accuracy = metrics.balanced_accuracy_score(epoch_labels, epoch_outputs)
print(accuracy)
