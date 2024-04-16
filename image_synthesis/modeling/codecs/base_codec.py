import torch
from torch import nn


class BaseCodec(nn.Module):
    
    def get_tokens(self, x, **kwargs):
        """
        Input: 
            x: input data
        Return:
            indices: B x L, the codebook indices, where L is the length 
                    of flattened feature map size
        """
        raise NotImplementedError

    def get_number_of_tokens(self):
        """
        Return: int, the number of tokens
        """
        raise NotImplementedError

    def encode(self, img):
        raise NotImplementedError

    def decode(self, img_seq):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            return super().train(True)
        else:
            return super().train(False)

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()


class DummyContentCodec(BaseCodec):
    '''
    This is just a dummy class, we don't tokenize the sketch. Instead, we let it pass as it is to embedder, ViT
    '''
    def __init__(self, token_shape, **kwargs):
        super().__init__()
        self.num_tokens = token_shape[0] * token_shape[1]
        self.trainable = False

    def get_tokens(self, x, **kwargs):
        return {'token': x}

    def get_number_of_tokens(self):
        """
        Return: int, the number of tokens
        """
        return self.num_tokens

    def check_length(self, token):
        return len(token) <= self.num_tokens

    def decode(self, x, *args, **kwargs):
        return x
