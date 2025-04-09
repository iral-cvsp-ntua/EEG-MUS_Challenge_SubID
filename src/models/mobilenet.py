import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        args_defaults=dict(
            in_channels=1, 
            num_classes=40, 
        )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        self.encoder = timm.create_model('tf_mobilenetv3_large_minimal_100', 
                                    pretrained=True,
                                    features_only=True)


        self.classifier2 = nn.Sequential(
            nn.Linear(960, self.num_classes), 
        )

        # self.emotion_classifier = nn.Sequential(
        #     nn.Linear(960, 2)
        # )


    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        # print('[X]', x.shape)
        features = self.encoder(x)[-1]

        # print('[FEATURES]', features.shape)

        features = F.avg_pool2d(features, features.size()[2:]).view(features.size(0), -1)

        # print('[FEATURES] - avg', features.shape)



        out = self.classifier2(features)
        # emotion = self.emotion_classifier(features)
        # print('[OUT]', out.shape)
        return out #, features
    
    def loss(self, probs,label):
        #print(probs,label)
        loss_func = nn.CrossEntropyLoss()
        #label = label.squeeze().argmax(dim=0)
        #print(probs.float(),label)
        return loss_func(probs,label)
    
    def windowed_loss(self, probs,label):
        loss_func = nn.CrossEntropyLoss()
        #print(len(label),len(probs))
        #if len(label) < len(probs):
        #    label = torch.concat((label,label[0]*torch.ones(9-len(label),dtype=torch.int).to(label.device)),dim=0)
        #print(len(label),len(probs))    
        return loss_func(probs, label)