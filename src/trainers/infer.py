import os
import torch
import src.models as models
import torch.nn as nn
import numpy as np

from tqdm.auto import tqdm
from src.saver import Saver
from sklearn import metrics

class Inference:

    def __init__(self, args):
        # Store args
        self.args = args
        # Create criterion
        self.criterion = nn.CrossEntropyLoss()
        # Create saver
        if not args.inference:
            self.saver = Saver(args.logdir, args.tag)
        
    def windowing(self, batch):
        "Use for validation and test"
        data = batch['eeg']
        
        # divide in chunks according to the model (crop size 1000)
        n_chunks = data.shape[2] // self.args.crop_size
        chunks = torch.split(data, self.args.crop_size, dim=2)
        #print(chunks)
        include_last = (data.shape[2] % self.args.crop_size) == 0
        if include_last:
            chunks = torch.cat(chunks, dim=0)
        else:
            chunks = torch.cat(chunks[:-1], dim=0)
        
        
        #print(chunks.shape)
        batch['eeg'] = chunks
        batch['label'] = batch['label'].repeat(n_chunks)
        
        return batch
        
    def train(self, loaders):

        self.attention = self.args.attention
        # Compute splits names
        splits = list(loaders.keys())

        splits = ['test']#'train','valid']#'test']#_trial']#,'val_subject']#,'test']
        # Setup model
        module = getattr(models, self.args.model)
        net = getattr(module, "Model")(vars(self.args))
        
        model_dict = net.state_dict()
        # Check resume
        if self.args.resume is not None:
            pretrained_dict = torch.load(self.args.resume)
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            net.load_state_dict(model_dict)




        # Move to device
        net.to(self.args.device)

        # Optimizer params
        optim_params = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.optimizer == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif self.args.optimizer == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}
        
        # Create optimizer
        optim_class = getattr(torch.optim, self.args.optimizer)
        optim = optim_class(params=[param for param in net.parameters() if param.requires_grad], **optim_params)

        # Configure scheduler
        if self.args.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optim, 
                mode = 'min', 
                patience=self.args.patience, 
                factor=self.args.reduce_lr_factor
            )
        else:
            scheduler = None

        # Initialize the final result metrics
        result_metrics = { split: {} for split in splits }

        # Train metrics
        lowest_train_loss = float('inf')
        
        # Validation metrics
        max_val_accuracy = -1
        max_val_accuracy_balanced = -1
        
        # Watch model if enabled
        if self.args.watch_model == True:
            self.saver.watch_model(net)

        # Process each epoch
        try:
            
            for split in splits:
        
                # Epoch metrics
                epoch_metrics = {}
                epoch_labels = []
                epoch_outputs = []


                net.eval()
                torch.set_grad_enabled(False)

                outs = []
                labs = []
                # Process each batch
                for batch in tqdm(loaders[split]):
                    batch = self.windowing(batch)
                        
                    # Get inputs and labels
                    inputs = batch['eeg']
                    labels = batch['label']

                    #print(torch.mean(inputs,axis=(1,2)),torch.std(inputs,axis=(1,2)))   
                    # Move to device
                    inputs = inputs.to(self.args.device)
                    labels = labels.to(self.args.device)

                    # Forward
                    outputs = net(inputs)
                    
                    # Check NaN
                    if torch.isnan(outputs).any():
                        raise FloatingPointError('Found NaN values')
                    
                    #print('prior to loss computation',outputs.shape,labels.shape)
                    if len(labels) < len(outputs):
                        labels = torch.concat((labels,labels[0]*torch.ones(9-len(labels),dtype=torch.int).to(labels.device)),dim=0)
                    elif len(labels) > len(outputs):
                        labels = labels[:9]
                    # Compute loss
                    loss = self.criterion(outputs, labels)
                   
                    outs.append(outputs)
                    labs.append(labels)
                    # Initialize metrics
                    batch_metrics = {
                        'loss': loss.item(),
                    }

                    if self.args.use_voting:
                        if self.args.voting_strategy == 'mean':
                            outputs = outputs.mean(dim=0)
                        elif self.args.voting_strategy == 'max':
                            outputs, _ = outputs.max(dim=0)
                        elif self.args.voting_strategy == 'min':
                            outputs, _ = outputs.min(dim=0)
                        elif self.args.voting_strategy == 'median':
                            outputs, _ = outputs.median(dim=0)
                        elif self.args.voting_strategy == 'majority':
                            try:
                                outputs = outputs.argmax(dim=1).mode().values[0]
                            except IndexError:
                                outputs = outputs.argmax(dim=1).mode().values
                        else:
                            raise ValueError(f"Voting strategy {self.args.voting_strategy} not recognized")

                        #print(outputs.shape)                            
                        outputs = outputs.unsqueeze(0)
                        labels = labels[0].unsqueeze(0)
                    
                    epoch_labels.append(labels)
                    epoch_outputs.append(outputs)
            

                    # Add metrics to epoch results
                    for k, v in batch_metrics.items():
                        v *= inputs.shape[0]
                        epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

                # Aggregate logits and labels
                epoch_labels = torch.cat(epoch_labels)
                epoch_outputs = torch.cat(epoch_outputs)
                print(self.args.splitnum, self.args.resume)
                np.save('outputs_'+self.args.run_name+'_'+self.args.splitnum+'.npy',epoch_outputs.cpu())
                if epoch_outputs.dim() > 1:
                    epoch_outputs = epoch_outputs.argmax(dim=1)

                np.save('labels.npy',epoch_labels.cpu())

               
        
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            pass

        except FloatingPointError as err:
            print(f'Error: {err}')
        
        
        return net, result_metrics
    
    def predict(self, test_loaders):
        
        # Setup model
        module = getattr(models, self.args.model)
        net = getattr(module, "Model")(vars(self.args))
        
        # Check resume
        if self.args.resume is not None:
            checkpoint = os.path.join(self.args.resume, f"{self.args.model}.pth")
            state_dict = torch.load(checkpoint)
            net.load_state_dict(state_dict)

        # Move to device
        net.to(self.args.device)
        
        # Set network mode
        net.eval()
        torch.set_grad_enabled(False)
        
        predictions = {}
        
        # Process each batch
        for split, test_loader in test_loaders.items():
            
            # Initialize predictions
            split_predictions = []
            
            outlist = []
            # Process each batch
            for batch in tqdm(test_loader):
                
                batch = self.windowing(batch)
                
                # Get inputs and labels
                inputs = batch['eeg']
                
                # Move to device
                inputs = inputs.to(self.args.device)
                
                # Forward
                outputs = net(inputs)
                
                outlist.append(outputs)

        return outlist

    def aggregate(self,outputs):
                
        predictions = {'test_trial': []}
        for output in outputs:
            print(output)
                # Check NaN
            if torch.isnan(output).any():
                raise FloatingPointError('Found NaN values')
            
            # Predictions
            if self.args.voting_strategy == 'mean':
                prediction = output.mean(dim=0).argmax()
            elif self.args.voting_strategy == 'max':
                prediction, _ = output.max(dim=0)
                prediction = prediction.argmax()
            elif self.args.voting_strategy == 'min':
                prediction, _ = output.min(dim=0)
                prediction = prediction.argmax()
            elif self.args.voting_strategy == 'median':
                prediction, _ = output.median(dim=0)
                prediction = prediction.argmax()
            elif self.args.voting_strategy == 'majority':
                try:
                    prediction = output.argmax(dim=1).mode().values[0]
                except IndexError:
                    prediction = output.argmax(dim=1).mode().values
            else:
                raise ValueError(f"Voting strategy {self.args.voting_strategy} not recognized")

            predictions['test_trial'].append(prediction.item())
            
        return predictions
