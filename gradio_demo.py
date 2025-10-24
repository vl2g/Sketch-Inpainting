import gradio as gr
import os
import numpy as np
import torch
from PIL import Image
from inference_VQ_Diffusion import VQ_Diffusion, load_yaml_config, build_dataloader
from torchvision import transforms
from functools import partial


def inpaint(image, sketch, truncation_rate):
    model = VQ_Diffusion
    try:
        image, mask = image['image'], image['mask']
        image = np.transpose(np.array(image.convert('RGB')).astype(np.float32), (2, 0, 1))
        sketch = np.transpose(np.array(sketch.convert('RGB')).astype(np.float32), (2, 0, 1))
        mask = np.array(mask.convert('L'))
        truncation_rate = truncation_rate / 100
        # print(mask)
        fast = False
        print(image.shape)
        print(mask.shape)
        print(sketch.shape)

        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'

        quantized_image = model.model.content_codec.vq.get_tokens(
            transforms.Resize((256, 256))(torch.tensor(image)).unsqueeze(0).cuda()
        )['token']
        print(quantized_image.shape)

        batch = {
            'quantized_image': quantized_image.cuda().repeat([SAMPLES_TO_GEN, 1]),
            'sketch': transforms.Resize((224, 224))(torch.tensor(sketch)).unsqueeze(0).cuda().repeat([SAMPLES_TO_GEN, 1, 1, 1]),
            'obj_mask': transforms.Resize((256, 256))(torch.tensor(np.where(mask > 10, 1., 0.)).unsqueeze(0)).unsqueeze(0).cuda().repeat([SAMPLES_TO_GEN, 1, 1, 1])
        }

        with torch.no_grad():
            model_out = model.model.generate_content(
                batch=batch,
                filter_ratio=0,
                replicate=1,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)

        return [Image.fromarray(content[i]) for i in range(SAMPLES_TO_GEN)]
    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/sketch_coco_inpainting.yaml', help='Path to config')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/000297e_1343979iter.pth', help='Path to checkpoint')
    args = parser.parse_args()

    SAMPLES_TO_GEN = 5

    config = load_yaml_config(args.config_path)
    print('Loading the model')
    VQ_Diffusion = VQ_Diffusion(config=args.config_path, path=args.ckpt_path)
    VQ_Diffusion.model.eval()

    demo = gr.Interface(fn=inpaint, 
             inputs=[
                gr.Image(type='pil', label='Input Image', tool='sketch'),
                gr.Image(type='pil', label='Sketch'),
                gr.Slider(10, 100, value=86)
             ],
             outputs=[gr.Image(type='pil', shape=(256, 256)) for _ in range(SAMPLES_TO_GEN)])

    demo.launch(debug=True)