# EEG-MUS_Challenge_SubID
IRAL-NTUA's team solutions for Track 1 of the EEG-Music Signal Processing GC, hosted @ICASSP 2025.

The EEG-Music Signal Processing Grand Challenge concerns 1) Person Identification and 2) Emotion Recognition from EEG recordings, collected from participants while listening to musical pieces. For the Person Identification Track, we propose a three-network ensemble, pre-trained through distinct datasets and (supervised or self-supervised) strategies: In-dataset contrastive self-supervised pretraining, in-domain supervised pretraining and out-of-domain supervised pretraining. This repository is built on the [official challenge repository](https://github.com/SalvoCalcagno/eeg-music-challenge-icassp-2025-baselines) and currently contains:

i) Weights for the contrastively-pretrained and DEAP-pretrained networks (the ImageNet weights are obtained through a [publicly available checkpoint](https://huggingface.co/docs/timm/en/models/mobilenet-v3), and are loaded in-code).

ii) Code for contrastive network pre-training, and

iii) Code for network finetuning on the training/validation split of the dataset (internally re-ordered into training, validation and testing data in a 4:1:1 ratio).

To run the contents of the repository, a functioning python environment with ```pytorch```, ```wandb``` and the ```timm``` package is required; alternatively, you can set up the necessary libraries with the provided ```.yml``` file.

For contrastive pre-training, enter the ```contrastive_setup``` sub-directory, and run the following command:

```python3 train.py --run_name [name of your run] --model [architecture_name] --augments [augmentation_list]```, where:

- ```--model``` defines the backbone architecture and can be either of ```{eegchannelnet, mobilenet}```

- ```--augment``` defines the augmentations to be applied at each contrastive pair before being fed to the network, and can be any among ```{crop,chanmask,timemask}```
  - ```crop```: cut two different 10-sec slices from the EEG
  - ```chanmask```: randomly mask out (hide) approx. half of the EEG channels.
  - ```timemask```: randomly mask out (hide) approx. half of the EEG timesteps.  

To fine-tune the models from the given-weights, run the following commands:

- Contrastive pre-training: ```python3 train.py --task subject_identification --split_dir data/splits/ --splitnum splitnum --model eegchannelnet --resume contrastive.pth --lr 0.0001```
  
- DEAP (in-domain) pre-training: ```python3 train.py --task subject_identification --split_dir data/splits/ --splitnum  splitnum --model mobilenet --resume mobilenet_deap.pth --lr 0.001```

- ImageNet (out-of-domain) pre-training: ```python3 train.py --task subject_identification --split_dir data/splits/ --splitnum splitnum --model mobilenet --lr 0.0001```

Splitnum corresponds to the number of cross-validation fold, and takes values in the range [0, 5].
