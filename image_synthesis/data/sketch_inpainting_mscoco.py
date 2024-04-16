import torch.utils.data as data
import albumentations as A
from torchvision import transforms
from PIL import Image
import cv2
import os
import torch
import numpy as np
import PIL
import json
from pathlib import Path
import random
import pickle
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    x, y, w, h = bbox       # MS-COCO style
    mask[y:y+h+1, x:x+w+1, :] = 1

    return mask

def mask2bbox(mask, scale_factor=1.2):
    """
    Converts a binary mask to bounding box coordinates in the format x_min, y_min, x_max, y_max.
    mask: c h w
    """
    extension = scale_factor - 1
    r, c = np.where(mask[0] == 1)
    ymin, ymax = min(r), max(r)
    xmin, xmax = min(c), max(c)

    if scale_factor > 1.:
        h = ymax - ymin + 1
        w = xmax - ymax + 1

        dh, dw = extension*h // 2, extension*w // 2

        ymin = max(0, ymin - dh)
        ymax = min(mask.shape[1], ymax + dh)

        xmin = max(0, xmin - dw)
        xmax = min(mask.shape[2], xmax + dw)


    return ymin, ymax, xmin, xmax

def cocoid2img(id):
    id = str(id)
    return '0'*(12-len(id)) + id

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')


class COCOSketchInpaintDataset(data.Dataset):
    def __init__(self, data_root, phase='train', image_size=[256, 256], sketch_size=[224, 224], sketch_subdir=None, image_subdir='data', data_len=-1, loader=pil_loader):
        # print(data_root)
        self.data_root = os.path.join(data_root, phase)
        self.sketch_subdir = sketch_subdir

        # imgs = make_dataset(os.path.join(data_root, 'data'))
        with open(os.path.join(self.data_root, 'bbox.json')) as f:
            self.bbox_ann = json.load(f)
        
        imgs = [os.path.join(self.data_root, image_subdir, f'{cocoid2img(x)}.jpg') for x in self.bbox_ann.keys()]

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        self.mask_rescale = transforms.Resize((image_size[0], image_size[1]))
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        
        self.sketch_tfs = transforms.Compose([
                transforms.Resize((sketch_size[0], sketch_size[1])),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        self.loader = loader
        self.mask_mode = None
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.loader(path)
        im_w, im_h = img.size
        img = self.mask_rescale(self.loader(path))

        image_name = Path(path).stem.split()[0]

        mask_idx, mask = self.get_mask(image_name, [im_h, im_w])
        mask = self.mask_rescale(mask)

        # mask_img = img*(1. - mask) + mask

        sketch = np.array(self.get_sketch(image_name, mask_idx)).astype(np.uint8)
        img = np.array(img).astype(np.uint8)
        ret['image'] = np.transpose(img.astype(np.float32), (2, 0, 1))
        # ret['mask_image'] = mask_img
        ret['obj_mask'] = mask[0]
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['sketch'] = np.transpose(sketch.astype(np.float32), (2, 0, 1))
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, image_name, img_size):
        bboxes = self.bbox_ann[image_name]
        mask_idx = random.randint(0, len(bboxes)-1)

        bbox = bboxes[mask_idx]

        mask = bbox2mask(img_size, bbox)

        return mask_idx, torch.from_numpy(mask).permute(2,0,1)
    
    def get_sketch(self, image_name, mask_idx):
        pth = os.path.join(self.data_root, self.sketch_subdir, f'{image_name}_{mask_idx}_out.png')
        sketch_img = self.loader(pth)
        sketch_img = self.sketch_tfs(sketch_img)

        return sketch_img


class COCOSketchInpaintDatasetQ(data.Dataset):
    def __init__(self, data_root, path_to_quantized='.', phase='train', image_size=[256, 256], sketch_size=[224, 224], sketch_subdir=None, image_subdir='data', data_len=-1, loader=pil_loader, **kwargs):
        # print(data_root)
        self.data_root = os.path.join(data_root, phase)
        self.sketch_subdir = sketch_subdir
        with open(os.path.join(path_to_quantized, f'coco_{phase}_vq_tokens.pkl'), 'rb') as f:
            self.quantized_images = pickle.load(f)


        # imgs = make_dataset(os.path.join(data_root, 'data'))
        with open(os.path.join(self.data_root, 'bbox.json')) as f:
            self.bbox_ann = json.load(f)
        
        imgs = [os.path.join(self.data_root, image_subdir, f'{cocoid2img(x)}.jpg') for x in self.bbox_ann.keys()]

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        self.mask_rescale = transforms.Resize((image_size[0], image_size[1]))
        
        self.sketch_tfs = transforms.Compose([
                transforms.Resize((sketch_size[0], sketch_size[1])),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        self.sketch_size = sketch_size[0]

        self.loader = loader
        self.mask_mode = None
        self.image_size = image_size
        self.phase = phase

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.loader(path)
        im_w, im_h = img.size
        img = self.mask_rescale(self.loader(path))

        image_name = Path(path).stem.split()[0]

        mask_idx, mask = self.get_mask(image_name, [im_h, im_w], img_idx=index)
        mask = self.mask_rescale(mask)
        while mask.max() == 0:
            # if the object gets lost in interpolation
            mask_idx, mask = self.get_mask(image_name, [im_h, im_w], img_idx=index)
            mask = self.mask_rescale(mask)

        # mask_img = img*(1. - mask) + mask

        sketch = np.array(self.get_sketch(image_name, mask_idx, (im_w, im_h))).astype(np.uint8)
        img = np.array(img).astype(np.uint8)
        ret['image'] = np.transpose(img.astype(np.float32), (2, 0, 1))
        # ret['mask_image'] = mask_img
        ret['obj_mask'] = mask[0]
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['sketch'] = np.transpose(sketch.astype(np.float32), (2, 0, 1))
        ret['quantized_image'] = np.array(self.quantized_images[image_name])
        ret['sketch_file'] = Path(path).stem.split('.')[0] + '_' + str(mask_idx)
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, image_name, img_size, img_idx=0):
        bboxes = self.bbox_ann[image_name]

        if self.phase == 'validation':
            random.seed(img_idx*5322)

        mask_idx = random.randint(0, len(bboxes)-1)

        bbox = bboxes[mask_idx]

        mask = bbox2mask(img_size, bbox)

        return mask_idx, torch.from_numpy(mask).permute(2,0,1)
    
    def get_sketch(self, image_name, mask_idx, orig_img_size):
        im_w, im_h = orig_img_size

        pth = os.path.join(self.data_root, self.sketch_subdir, f'{image_name}_{mask_idx}_out.png')
        sketch_img = self.loader(pth)

        sketch_w, sketch_h = sketch_img.size

        scale_w, scale_h = self.image_size[1] / im_w, self.image_size[0] / im_h
        new_w, new_h = round(scale_w*sketch_w), round(scale_h*sketch_h)

        s = max(new_h, new_w)
        r = self.sketch_size / s
        s = (round(r * new_h), round(r * new_w))
        sketch_img = np.array(TF.resize(sketch_img, s)).astype(np.uint8)

        return A.PadIfNeeded(self.sketch_size, self.sketch_size, always_apply=True,
                             border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255])(image=sketch_img)['image']


class COCOSketchInpaintDatasetQSketchList(data.Dataset):
    def __init__(self, data_root, path_to_quantized='.', phase='train', image_size=[256, 256], sketch_size=[224, 224], sketch_subdir=None, image_subdir='data', data_len=-1, loader=pil_loader, sketch_list_file='/DATA/nakul/sketch/SketchInpDiffusion/scripts/correctly_classified_sketches.txt', **kwargs):
        # print(data_root)
        self.data_root = os.path.join(data_root, phase)
        self.sketch_subdir = sketch_subdir
        with open(os.path.join(path_to_quantized, f'coco_{phase}_vq_tokens.pkl'), 'rb') as f:
            self.quantized_images = pickle.load(f)


        # imgs = make_dataset(os.path.join(data_root, 'data'))
        with open(os.path.join(self.data_root, 'bbox.json')) as f:
            self.bbox_ann = json.load(f)
        
        with open(sketch_list_file) as f:
            self.sketches_list = f.readlines()
        self.sketches_list = [x.strip() for x in self.sketches_list]

        self.mask_rescale = transforms.Resize((image_size[0], image_size[1]))
        
        self.sketch_tfs = transforms.Compose([
                transforms.Resize((sketch_size[0], sketch_size[1])),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        self.sketch_size = sketch_size[0]

        self.loader = loader
        self.mask_mode = None
        self.image_size = image_size
        self.phase = phase

    def __getitem__(self, index):
        ret = {}
        f, bbox_idx = self.sketches_list[index].split('_')[:2]
        bbox_idx = int(bbox_idx)
        path = os.path.join(self.data_root, 'data', f+'.jpg')
        img = self.loader(path)
        im_w, im_h = img.size
        img = self.mask_rescale(self.loader(path))

        image_name = Path(path).stem.split()[0]

        mask_idx, mask = self.get_mask(image_name, [im_h, im_w], mask_idx=bbox_idx, img_idx=index)
        mask = self.mask_rescale(mask)

        # mask_img = img*(1. - mask) + mask

        sketch = np.array(self.get_sketch(image_name, bbox_idx, (im_w, im_h))).astype(np.uint8)
        img = np.array(img).astype(np.uint8)
        ret['image'] = np.transpose(img.astype(np.float32), (2, 0, 1))
        # ret['mask_image'] = mask_img
        ret['obj_mask'] = mask[0]
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['sketch'] = np.transpose(sketch.astype(np.float32), (2, 0, 1))
        ret['quantized_image'] = np.array(self.quantized_images[image_name])
        ret['sketch_file'] = self.sketches_list[index]
        return ret

    def __len__(self):
        return len(self.sketches_list)

    def get_mask(self, image_name, img_size, mask_idx, img_idx=0):
        bboxes = self.bbox_ann[image_name]

        if self.phase == 'validation':
            random.seed(img_idx*5322)

        bbox = bboxes[mask_idx]

        mask = bbox2mask(img_size, bbox)

        return mask_idx, torch.from_numpy(mask).permute(2,0,1)
    
    def get_sketch(self, image_name, mask_idx, orig_img_size):
        im_w, im_h = orig_img_size

        pth = os.path.join(self.data_root, self.sketch_subdir, f'{image_name}_{mask_idx}_out.png')
        sketch_img = self.loader(pth)

        sketch_w, sketch_h = sketch_img.size

        scale_w, scale_h = self.image_size[1] / im_w, self.image_size[0] / im_h
        new_w, new_h = round(scale_w*sketch_w), round(scale_h*sketch_h)

        s = max(new_h, new_w)
        r = self.sketch_size / s
        s = (round(r * new_h), round(r * new_w))
        sketch_img = np.array(TF.resize(sketch_img, s)).astype(np.uint8)

        return A.PadIfNeeded(self.sketch_size, self.sketch_size, always_apply=True,
                             border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255])(image=sketch_img)['image']



class COCOSketchInpaintDatasetQwithBBOX(data.Dataset):
    def __init__(self, data_root, path_to_quantized='.', phase='train', image_size=[256, 256],
                 sketch_size=[224, 224], sketch_subdir=None, image_subdir='data', data_len=-1,
                 loader=pil_loader, lpips_bbox_scale_factor=1., **kwargs):
        # print(data_root)
        self.data_root = os.path.join(data_root, phase)
        self.sketch_subdir = sketch_subdir
        with open(os.path.join(path_to_quantized, f'coco_{phase}_vq_tokens.pkl'), 'rb') as f:
            self.quantized_images = pickle.load(f)


        # imgs = make_dataset(os.path.join(data_root, 'data'))
        with open(os.path.join(self.data_root, 'bbox.json')) as f:
            self.bbox_ann = json.load(f)
        
        imgs = [os.path.join(self.data_root, image_subdir, f'{cocoid2img(x)}.jpg') for x in self.bbox_ann.keys()]

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

        self.mask_rescale = transforms.Resize((image_size[0], image_size[1]))
        
        # load a sketch, resize the largest size to be 224 maintaining the aspect ratio and paste it onto a blank canvas
        # of 224x224
        # self.sketch_tfs = A.Compose([
        #         A.LongestMaxSize(max_size=max(sketch_size), always_apply=True),
        #         A.PadIfNeeded(sketch_size[0], sketch_size[1], always_apply=True,
        #                       border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # ])

        self.sketch_tfs = transforms.Compose([
                transforms.Resize((sketch_size[0], sketch_size[1])),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        self.lpips_bbox_scale_factor = lpips_bbox_scale_factor

        self.loader = loader
        self.mask_mode = None
        self.image_size = image_size
        self.phase = phase

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.loader(path)
        im_w, im_h = img.size
        img = self.mask_rescale(self.loader(path))

        image_name = Path(path).stem.split()[0]

        mask_idx, mask = self.get_mask(image_name, [im_h, im_w], img_idx=index)
        mask = self.mask_rescale(mask)

        while mask.max() == 0:
            # if the object gets lost in interpolation
            mask_idx, mask = self.get_mask(image_name, [im_h, im_w], img_idx=index)
            mask = self.mask_rescale(mask)

        ymin, ymax, xmin, xmax = mask2bbox(mask, scale_factor=self.lpips_bbox_scale_factor)
        roi_bbox = [xmin, ymin, xmax, ymax]


        # mask_img = img*(1. - mask) + mask

        sketch = np.array(self.get_sketch(image_name, mask_idx)).astype(np.uint8)
        img = np.array(img).astype(np.uint8)
        ret['image'] = np.transpose(img.astype(np.float32), (2, 0, 1))
        # ret['mask_image'] = mask_img
        ret['obj_mask'] = mask[0]
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['sketch'] = np.transpose(sketch.astype(np.float32), (2, 0, 1))
        ret['quantized_image'] = np.array(self.quantized_images[image_name])
        ret['roi_bbox'] = np.array(roi_bbox).astype(np.float)

        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self, image_name, img_size, img_idx=0):
        bboxes = self.bbox_ann[image_name]

        if self.phase == 'validation':
            random.seed(img_idx*5322)

        mask_idx = random.randint(0, len(bboxes)-1)

        bbox = bboxes[mask_idx]

        mask = bbox2mask(img_size, bbox)

        return mask_idx, torch.from_numpy(mask).permute(2,0,1)
    
    def get_sketch(self, image_name, mask_idx):
        # pth = os.path.join(self.data_root, self.sketch_subdir, f'{image_name}_{mask_idx}_out.png')
        # sketch_img = self.loader(pth)
        # sketch_img = self.sketch_tfs(image=np.array(sketch_img))['image']

        # return sketch_img

        pth = os.path.join(self.data_root, self.sketch_subdir, f'{image_name}_{mask_idx}_out.png')
        sketch_img = self.loader(pth)
        sketch_img = self.sketch_tfs(sketch_img)

        return sketch_img