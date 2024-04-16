import gradio as gr
import os
import numpy as np
import torch
from PIL import Image
from inference_VQ_Diffusion import VQ_Diffusion, load_yaml_config, build_dataloader
from torchvision import transforms
from functools import partial


SAMPLES_TO_GEN = 5

base_pth = '/DATA/nakul/sketch/SketchInpDiffusion/OUTPUT/sketch_inp_diff'
config_pth = os.path.join(base_pth, 'configs', 'config.yaml')
config = load_yaml_config(config_pth)
print('Loading the model')
VQ_Diffusion = VQ_Diffusion(config=config_pth, path=os.path.join(base_pth, 'checkpoint', '/DATA/nakul/sketch/SketchInpDiffusion/OUTPUT/sketch_inp_diff/checkpoint/000297e_1343979iter.pth'))
VQ_Diffusion.model.eval()

def inpaint(image, sketch, truncation_rate):
    model = VQ_Diffusion
    print('Hereee')
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

demo = gr.Interface(fn=inpaint, 
             inputs=[
                gr.Image(type='pil', label='Input Image', tool='sketch'),
                gr.Image(type='pil', label='Sketch'),
                gr.Slider(10, 100, value=86)
             ],
             outputs=[gr.Image(type='pil', shape=(256, 256)) for _ in range(SAMPLES_TO_GEN)])

if __name__ == '__main__':

    demo.launch(debug=True, server_name="10.6.0.40", server_port=8000)