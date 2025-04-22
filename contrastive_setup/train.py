import os
import numpy as np
import json
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import EEGChunkedDataset
from model import ContrastiveModel, ContrastiveMobileNet
import argparse  


def create_run_directory(run_name):
    base_dir = 'checkpoints'
    os.makedirs(base_dir, exist_ok=True)
    
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


parser = argparse.ArgumentParser(description="EEG-Wav Contrastive Learning")
parser.add_argument('--run_name', type=str, required=True, help="Name of the current run (used for saving checkpoints)")
parser.add_argument('--model',type=str,default='eegchannelnet',required=False)
parser.add_argument('--augments',nargs='+',default=['crop'])
args = parser.parse_args()

print(args.augments)
run_dir = create_run_directory(args.run_name)
print(f"Saving checkpoints in directory: {run_dir}")

splits = json.load(open("data/splits/splits_subject_identification.json"))

keys = []
for split in splits:
    for trial in splits[split]:
        keys.append([split.replace('val_trial','train').replace('val_subject','train'), trial["id"]])

print(len(keys))

random.shuffle(keys)

train_tuples = keys[:int(0.8 * len(keys))]
valid_tuples = keys[int(0.8 * len(keys)):]

print(f'Training tuples: {len(train_tuples)}')
print(f'Validation tuples: {len(valid_tuples)}')

BATCH_SIZE = 8

train_dataset = EEGChunkedDataset(train_tuples, augments = args.augments, data_type='preprocessed', segment_size_seconds=10, layer=-4)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

valid_dataset = EEGChunkedDataset(valid_tuples, augments = args.augments, data_type='preprocessed', segment_size_seconds=10, layer=-4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

OUTPUT_DIM = 256

if args.model == 'eegchannelnet':
    model = ContrastiveModel()
elif args.model == 'mobilenet':
    model = ContrastiveMobileNet()
else:
    print('Model not supported!')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EPOCHS = 200
save_counter = 0
best_val_loss = float('inf')  

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for batch_idx, (eeg_data, wav_data) in train_progress:
        eeg_data = eeg_data.to(device)
        wav_data = wav_data.to(device)


        optimizer.zero_grad()

        eeg_features, wav_features = model(eeg_data, wav_data)

        #print(eeg_features[0,:10],wav_features[0,:10])
        loss,acc = model.contrastive_loss(eeg_features, wav_features)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        total_acc += acc.item()
        train_progress.set_postfix({'Batch Loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc/ len(train_tuples)
    print('train loss', avg_loss, 'train acc', avg_acc)

    
    model.eval()
    total_val_loss = 0.0
    total_val_acc = 0.0

    val_progress = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validation {epoch + 1}/{EPOCHS}")

    with torch.no_grad():
        for batch_idx, (eeg_data, wav_data) in val_progress:
            eeg_data = eeg_data.to(device)
            wav_data = wav_data.to(device)

            eeg_features, wav_features = model(eeg_data, wav_data)

            val_loss,val_acc = model.contrastive_loss(eeg_features, wav_features)
       
            total_val_loss += val_loss.item()
            total_val_acc += val_acc.item()

            val_progress.set_postfix({'Val Batch Loss': f'{val_loss.item():.4f}'})

    avg_val_loss = total_val_loss / len(valid_loader)
    avg_val_acc = total_val_acc / len(valid_tuples)
    print('val loss', avg_val_loss, 'val acc', avg_val_acc)


    tqdm.write(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        save_counter += 1
        best_val_loss = avg_val_loss
        if save_counter == 2:  # Save model every 5th validation improvement
            model_path = os.path.join(run_dir, f'contrastive_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"Model saved as {model_path}")
            save_counter = 0  
    
    #print('k')