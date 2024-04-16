from image_synthesis.modeling.codecs.image_codec.taming_gumbel_vqvae import TamingGumbelVQVAE
from image_synthesis.data.sketch_inpainting_mscoco import COCOSketchInpaintDataset
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import tqdm

if __name__ == '__main__':
    device = 'cuda'
    phase = 'validation'
    ds = COCOSketchInpaintDataset(data_root='../../fiftyone/coco-2017/', sketch_subdir='contours_seg', phase=phase)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    codec = TamingGumbelVQVAE().to(device)
    vq_tokens = {}

    with torch.no_grad():
        for data in tqdm.tqdm(dl):
            img = data['image'].to(device)
            tokens = codec.get_tokens(img)['token'].detach().cpu()
            paths = [Path(x).stem.split('.')[0] for x in data['path']]

            for i, p in enumerate(paths):
                vq_tokens[p] = tokens[i]
    

    torch.save(vq_tokens, f'coco_{phase}_vq_tokens.pt')