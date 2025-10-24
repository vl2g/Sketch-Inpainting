<h1 align="center">
Sketch-guided Image Inpainting with Partial Discrete Diffusion Process
</h1>


[![arXiv](https://img.shields.io/badge/arXiv-2404.11949-b31b1b.svg)](https://arxiv.org/abs/2404.11949)





<hr>

Official Implementation of the Sketch-guided Image Inpainting with Partial Discrete Diffusion Process paper (CVPR-W 2024)

## Setting up the Environment
We provide `env.yml` file to setup the conda environment. Please install it as follows:

```bash
conda env create -f env.yml
conda activate sketch
```

## Downloading the Data
Please download the sketches and accompanying data from (here)[] and extract the folder into the repo's root directory. The tokenized images for train and validation splits can be downloaded from (here)[] --- these should also be placed in the root directory (if downloading at other location, please make sure to update the paths in `configs/sketch_coco_inpainting.yaml`).


After this, please download `train` and `val` splits of the MS-COCO images, extract them and put them under `data/train/images` and `data/validation/images`, respectively.



The directory folder for `data` directory should now look like:

```
data
|----train
|    |----contours_seg_resized
|    |----images
|    |----bbox_annotations.json
|    |----bbox.json
|    |----labels.json
|
|----validation
|    |----contours_seg_resized
|    |----images
|    |----bbox_annotations.json
|    |----bbox.json
|    |----labels.json
|
|----info.json
```


## Training the Model

Begin by downloading the VQ-VAE `openimages-f8-8192` from [here](https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt) and put it under `checkpoints/ckeckpoints/taming_f8_8192_openimages_last.pth`.

To start the training, run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --config_file configs/sketch_coco_inpainting.yaml\
    --name sketch_inp_diff \
    --num_node 1
```

The pre-trained model can be downloaded from [here]().

## Launching the Gradio Demo
After downloading the VQ-VAE as instructed in previous section, please run the following command to launch a Gradio Demo:
```bash
python gradio_demo.py --config_path configs/sketch_coco_inpainting.yaml \
    --ckpt_path <PATH_TO_CHECKPOINT>
```

## Cite Us
If you find our work helpful, please consider citing us:
```
@InProceedings{Sharma_2024_CVPR,
    author    = {Sharma, Nakul and Tripathi, Aditay and Chakraborty, Anirban and Mishra, Anand},
    title     = {Sketch-guided Image Inpainting with Partial Discrete Diffusion Process},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {6024-6034}
}
```

## Acknowledgements
We would like to thank the authors of [VQ-Diffusion](https://github.com/microsoft/VQ-Diffusion) and [Taming Transformers](https://github.com/CompVis/taming-transformers) for open-sourcing their code and checkpoints!
