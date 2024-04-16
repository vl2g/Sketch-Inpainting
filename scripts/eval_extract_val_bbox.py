import json
import os
import tqdm
import cv2
from PIL import Image
if __name__ == '__main__':

    base_dir = '/DATA/nakul/sketch/SketchInpDiffusion/RESULT/final_inf/out'
    bbox_dir = os.path.join(base_dir, 'bbox')

    bbox_file = '../image_synthesis/data/eval_bbox_data.json'
    with open(bbox_file) as f:
        bbox_data = json.load(f)
    try:
        os.makedirs(bbox_dir)
    except Exception as e:
        print(e)

    
    for f in tqdm.tqdm(os.listdir(os.path.join(base_dir))):
        file_pth = os.path.join(base_dir, f)

        if os.path.isdir(file_pth):
            continue
        bbox = bbox_data[f.split('.')[0].split('_')[0]]

        img = cv2.imread(file_pth)
        # Get bounding box coordinates
        ymin, ymax, xmin, xmax = bbox
        # Crop image
        cropped_img = img[ymin:ymax+1, xmin:xmax+1]
    
        try:
            cv2.imwrite(os.path.join(bbox_dir, f), cropped_img)
        except Exception as e:
            print(file_pth)
            print(e)
    



