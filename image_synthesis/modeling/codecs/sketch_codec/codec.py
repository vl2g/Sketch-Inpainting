from image_synthesis.modeling.codecs.base_codec import BaseCodec

class SketchCodec(BaseCodec):
    '''
    This is just a dummy class, we don't tokenize the sketch. Instead, we let it pass as it is to embedder, ViT
    '''
    def __init__(self, sketch_size=[256, 256], patch_size=[16, 16], **kwargs):
        super().__init__()
        self.num_tokens = (sketch_size[0] // patch_size[0]) * (sketch_size[1] // patch_size[1]) + 1 #[CLS]
        self.trainable = False

    def get_tokens(self, x, **kwargs):
        return x

    def get_number_of_tokens(self):
        """
        Return: int, the number of tokens
        """
        return self.num_tokens

    def check_length(self, token):
        return len(token) <= self.num_tokens