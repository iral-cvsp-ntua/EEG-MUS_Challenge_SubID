import mne
import os
import numpy as np
import torch
from torch import nn 
import random
from torch.utils.data import Dataset, DataLoader  

# set them as true/false
# to apply the respective contrastive augmentations

#crop = True
#chanmask = False
#timemask = False

class EEGChunkedDataset(Dataset): #to be used for EEG<->EEG cl
    def __init__(self, tuples, augments,data_type, segment_size_seconds=10, transform=None, layer=-1):
        self.tuples = tuples 
        self.data_type = data_type
        self.segment_size_seconds = segment_size_seconds
        self.eeg_sampling_rate = 128  # Hz for EEG
        self.embedding_time_resolution = 0.02  # 20ms for embeddings
        self.transform = transform
        self.layer = layer
        self.crop = 'crop' in augments
        self.chanmask = 'chanmask' in augments
        self.timemask = 'timemask' in augments

    def __len__(self):
        return len(self.tuples)
    
    def load_part_of_npy(self, filepath, slice_range, layer=None):
        embs = np.load(filepath, mmap_mode='r')  
        if layer is not None:
            return embs[layer, slice_range].squeeze()  
        return embs[slice_range].squeeze()  

    def load_part_of_fif(self, filepath, start_idx, end_idx):
        eeg_data = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        eeg_data = eeg_data.get_data()  
        return eeg_data[:, start_idx:end_idx]  

    def __getitem__(self, idx):
        sample = self.tuples[idx]

        subdir = self.data_type
        ext = "fif" if self.data_type in ["raw", "pruned"] else "npy"

        eeg_path = os.path.join('/gpu-data3/eeg_challenge/eremus_dataset/', subdir, sample[0], f"{sample[1]}_eeg.{ext}")

        if ext == "npy":
            eeg_data = np.load(eeg_path, mmap_mode='r')
        elif ext == "fif":
            eeg_data = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False).get_data()

        #eeg_total_time = eeg_data.shape[1] / self.eeg_sampling_rate  # Total EEG time in seconds

        # Decide which is shorter in terms of available time
        
        if self.chanmask: #pick an (approx. 50%) of channels to mask
            maska = np.random.randn(32) > 0
            maskb = np.random.randn(32) > 0

        segment_size_samples_eeg = int(self.segment_size_seconds * self.eeg_sampling_rate)
        if self.timemask: #create an unstructured temporal mask; 50% of timesteps are skipeed
            maskta = np.random.randn(segment_size_samples_eeg) > 0
            masktb = np.random.randn(segment_size_samples_eeg) > 0

        max_start_idx_eeg = eeg_data.shape[1] - segment_size_samples_eeg
        start_idx_eeg = random.randint(0, max_start_idx_eeg) #crop a random 10-sec excerpt
        end_idx_eeg = start_idx_eeg + segment_size_samples_eeg
        eeg_chunka = np.copy(eeg_data[:,start_idx_eeg:end_idx_eeg])

        if self.crop:
            start_idx_eeg = random.randint(0, max_start_idx_eeg) #pick a different excerpt for the second eeg
            end_idx_eeg = start_idx_eeg + segment_size_samples_eeg
        #eeg_chunkb = eeg_data[:, start_idx_eeg:end_idx_eeg]
        eeg_chunkb = np.copy(eeg_data[:,start_idx_eeg:end_idx_eeg])

        eeg_chunka = self.normalize_eeg(eeg_chunka)
        
        if self.chanmask:
            eeg_chunka[maska,:] = 0
        if self.timemask:
            eeg_chunka[:,maskta] = 0
        eeg_chunka = torch.tensor(eeg_chunka, dtype=torch.float32)

        eeg_chunkb = self.normalize_eeg(eeg_chunkb)
        if self.chanmask:
            eeg_chunkb[maskb,:] = 0
        if self.timemask:
            eeg_chunkb[:,masktb] = 0
        eeg_chunkb = torch.tensor(eeg_chunkb, dtype=torch.float32)

        return eeg_chunka, eeg_chunkb

    def normalize_eeg(self, eeg_chunk):
        mean = eeg_chunk.mean(axis=1, keepdims=True)
        std = eeg_chunk.std(axis=1, keepdims=True)
        eeg_chunk = (eeg_chunk - mean) / std  
        return eeg_chunk

class EEGWavChunkedDataset(Dataset):
    def __init__(self, tuples, data_type, segment_size_seconds=10, transform=None, layer=-1):
        self.tuples = tuples 
        self.data_type = data_type
        self.segment_size_seconds = segment_size_seconds
        self.eeg_sampling_rate = 128  # Hz for EEG
        self.embedding_time_resolution = 0.02  # 20ms for embeddings
        self.transform = transform
        self.layer = layer

    def __len__(self):
        return len(self.tuples)
    
    def load_part_of_npy(self, filepath, slice_range, layer=None):
        embs = np.load(filepath, mmap_mode='r')  
        if layer is not None:
            return embs[layer, slice_range].squeeze()  
        return embs[slice_range].squeeze()  

    def load_part_of_fif(self, filepath, start_idx, end_idx):
        eeg_data = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        eeg_data = eeg_data.get_data()  
        return eeg_data[:, start_idx:end_idx]  

    def __getitem__(self, idx):
        sample = self.tuples[idx]

        subdir = self.data_type
        ext = "fif" if self.data_type in ["raw", "pruned"] else "npy"

        eeg_path = os.path.join('/gpu-data3/eeg_challenge/eremus_dataset/', subdir, sample[0], f"{sample[2]}_eeg.{ext}")
        emb_filename = [f for f in os.listdir('/gpu-data3/eeg_challenge/mert_embeddings') if f.startswith(sample[1])][0]
        emb_path = os.path.join('/gpu-data3/eeg_challenge/mert_embeddings', emb_filename)

        if ext == "npy":
            eeg_data = np.load(eeg_path, mmap_mode='r')
        elif ext == "fif":
            eeg_data = mne.io.read_raw_fif(eeg_path, preload=True, verbose=False).get_data()
        
        emb_data = np.load(emb_path, mmap_mode='r')

        eeg_total_time = eeg_data.shape[1] / self.eeg_sampling_rate  # Total EEG time in seconds
        emb_total_time = emb_data.shape[1] * self.embedding_time_resolution  # Total Wav embedding time in seconds

        # Decide which is shorter in terms of available time
        if eeg_total_time <= emb_total_time:
            # EEG is shorter, so we'll base the chunk on EEG
            segment_size_samples_eeg = int(self.segment_size_seconds * self.eeg_sampling_rate)
            max_start_idx_eeg = eeg_data.shape[1] - segment_size_samples_eeg
            start_idx_eeg = random.randint(0, max_start_idx_eeg)
            end_idx_eeg = start_idx_eeg + segment_size_samples_eeg

            eeg_chunk = eeg_data[:, start_idx_eeg:end_idx_eeg]

            start_time = start_idx_eeg / self.eeg_sampling_rate
            end_time = end_idx_eeg / self.eeg_sampling_rate
            start_idx_emb = int(start_time / self.embedding_time_resolution)
            end_idx_emb = int(end_time / self.embedding_time_resolution)

            emb_chunk = emb_data[self.layer, start_idx_emb:end_idx_emb]

        else:
            # Wav embeddings are shorter, so we'll base the chunk on embeddings
            segment_size_emb_steps = int(self.segment_size_seconds / self.embedding_time_resolution)
            max_start_idx_emb = emb_data.shape[1] - segment_size_emb_steps
            start_idx_emb = random.randint(0, max_start_idx_emb)
            end_idx_emb = start_idx_emb + segment_size_emb_steps

            emb_chunk = emb_data[self.layer, start_idx_emb:end_idx_emb]


            start_time = start_idx_emb * self.embedding_time_resolution
            end_time = end_idx_emb * self.embedding_time_resolution
            start_idx_eeg = int(start_time * self.eeg_sampling_rate)
            end_idx_eeg = int(end_time * self.eeg_sampling_rate)

            eeg_chunk = eeg_data[:, start_idx_eeg:end_idx_eeg]    

        emb_chunk = np.mean(emb_chunk, axis=0)

        # Normalize EEG and convert to tensor
        eeg_chunk = self.normalize_eeg(eeg_chunk)
        eeg_chunk = torch.tensor(eeg_chunk, dtype=torch.float32)

        emb_chunk = torch.tensor(emb_chunk, dtype=torch.float32)

        return eeg_chunk, emb_chunk


    def normalize_eeg(self, eeg_chunk):
        mean = eeg_chunk.mean(axis=1, keepdims=True)
        std = eeg_chunk.std(axis=1, keepdims=True)
        eeg_chunk = (eeg_chunk - mean) / std  
        return eeg_chunk

