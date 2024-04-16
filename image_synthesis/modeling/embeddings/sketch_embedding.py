import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import ViTModel
from image_synthesis.modeling.modules.clip import clip
from image_synthesis.modeling.modules.clip import model as clip_model
from torchvision import models
from .base_embedding import BaseEmbedding

# class ViTEmbedding(BaseEmbedding):

#     def __init__(self, name='google/vit-base-patch16-224-in21k', trainable=True,
#                  embed_dim=512, layers_to_keep=6,
#                  use_avg=False, *args, **kwargs):

#         super(ViTEmbedding, self).__init__()
        
#         self.base_model = ViTModel.from_pretrained(name)
#         self.embed_dim = embed_dim
#         self.use_avg = use_avg

#         if layers_to_keep > 0:
#             self.base_model.encoder.layer = self.base_model.encoder.layer[:layers_to_keep]
        
        
#         self.proj = nn.Linear(
#             self.base_model.encoder.layer[-1].output.dense.out_features, embed_dim
#         )

#         self.trainable = trainable
#         self._set_trainable()

    
#     def forward(self, x):
#         x = self.base_model(x)['last_hidden_state']
#         x = self.proj(x)

#         if self.use_avg:
#             x = torch.mean(x[:, 1:], dim=1)
#         # x = F.normalize(x, -1)

#         return x


class ResNetEmbedding(BaseEmbedding):
    def __init__(self, name='resnet50', embed_dim=512, use_avg=False, pretrained_model_path='', trainable=True, *args, **kwargs):
        super(ResNetEmbedding, self).__init__()
        self.base_model = getattr(models, name)()

        if pretrained_model_path.endswith('pt'):
            self.load_base_model(pretrained_model_path)

        self.base_model.avgpool = nn.Identity()
        self.base_model.fc = nn.Identity()
        conv_out_features = self.base_model.layer4[-1].conv1.in_channels

        self.proj = nn.Linear(
            conv_out_features, embed_dim
        )

        self.embed_dim = embed_dim
        self.use_avg = use_avg
        self.trainable = trainable

        self.height_emb = nn.Embedding(7, embed_dim) # height   resnet will give 7x7 embedding
        self.width_emb = nn.Embedding(7, embed_dim)

        self._set_trainable()
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.base_model(x)
        x = x.view(x.shape[0], -1, self.proj.in_features)
        x = self.proj(x)

        if self.use_avg:
            x = torch.mean(x, dim=1)
        
        height_emb = self.height_emb(torch.arange(7, device=x.device).view(1, 7)).unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
        width_emb = self.width_emb(torch.arange(7, device=x.device).view(1, 7)).unsqueeze(1) 

        pos_emb = (height_emb + width_emb).view(1, 7 * 7, -1) # 1 x H x W x D -> 1 x L xD

        x = x + pos_emb[:, :x.shape[1], :]

        return x
    
    @torch.no_grad()
    def preprocess(self, x):
        return 1. - x/255.

    def load_base_model(self, pth):
        #TODO
        pass