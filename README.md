# EEG-MUS_Challenge_SubID
IRAL-NTUA's team solutions for Track 1 of the EEG-Music Signal Processing GC, hosted @ICASSP 2025.

The EEG-Music Signal Processing Grand Challenge concerns 1) Person Identification and 2) Emotion Recognition from EEG recordings, collected from participants while listening to musical pieces. For the Person Identification Track, we propose a three-network ensemble, pre-trained through distinct datasets and (supervised or self-supervised) strategies: In-dataset contrastive self-supervised pretraining, in-domain supervised pretraining and out-of-domain supervised pretraining. This repository is built on the [official challenge repository](https://github.com/SalvoCalcagno/eeg-music-challenge-icassp-2025-baselines) and contains:

i) Code for contrastive network pre-training, 

ii) Code for network finetuning on the training/validation split of the dataset (internally re-ordered into training, validation and testing data in a 4:1:1 ratio),

iii) Weights for the contrastively-pretrained and DEAP-pretrained networks (the ImageNet weights are obtained through a [publicly available checkpoint](https://huggingface.co/docs/timm/en/models/mobilenet-v3), and are loaded in-code),

and iv) Code for performing inference on unseen data, on the above training/validation split.

To run the contents of the repository, a functioning python environment with ```pytorch```, ```wandb``` and the ```timm``` package is required; alternatively, you can set up the necessary libraries with the provided ```.yml``` file.

For contrastive pre-training, enter the ```contrastive_setup``` sub-directory, and run the following command:

```python3 train.py --run_name [name of your run] --model [architecture_name] --augments [augmentation_list]```, where:

- ```--model``` defines the backbone architecture and can be either of ```{eegchannelnet, mobilenet}```

- ```--augment``` defines the augmentations to be applied at each contrastive pair before being fed to the network, and can be any among ```{crop,chanmask,timemask}```
  - ```crop```: cut two different 10-sec slices from the EEG
  - ```chanmask```: randomly mask out (hide) approx. half of the EEG channels.
  - ```timemask```: randomly mask out (hide) approx. half of the EEG timesteps.

The weights are saved in the subdirectory specified by the ```run_name`` argument (```checkpoints/run_name```).

To fine-tune the models from the given-weights, run the following commands:

- Contrastive pre-training: ```python3 train.py --task subject_identification --split_dir data/splits/ --splitnum splitnum --model eegchannelnet --resume contrastive.pth --lr 0.0001```
  
- DEAP (in-domain) pre-training: ```python3 train.py --task subject_identification --split_dir data/splits/ --splitnum  splitnum --model mobilenet --resume mobilenet_deap.pth --lr 0.001```

- ImageNet (out-of-domain) pre-training: ```python3 train.py --task subject_identification --split_dir data/splits/ --splitnum splitnum --model mobilenet --lr 0.0001```

The argument ```splitnum``` corresponds to the number of cross-validation fold, and takes values in the range [0, 5].

The weights are saved into the (automatically created if it does not already exist) ```exps/subject_identification/{model}/baseline_{timestamp}/{model}.pth``` subdirectory.

To acquire labels for a specific split, run the ```inference.py``` script as: ```python3 inference.py --run_name {my_run} --task subject_identification --split_dir data/splits --splitnum splitnum --model {eegchannelnet, mobilenet} --resume {path_to_model}```. The predicted labels for the specific model will be output at the ```output_{my_run}_{splitnum}.npy``` file. 

You can then run the ```estimate_score.py```, either for a single model or a model ensemble, on a pre-defined split, as:

```python3 estimate_score.py run_name1 run_name2 ... run_nameN splitnum```.

You can either provide a single run-name to evaluate a single model, or multiple ones to evaluate a model ensemble -- these should correspond to run-names provided as ```run_name``` arguments of ```inference.py```. 
