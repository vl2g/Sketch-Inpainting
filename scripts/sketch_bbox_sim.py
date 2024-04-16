
from simclrv2.resnet import get_resnet, name_to_params
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SketchDataset(Dataset):
    def __init__(self, sketch_dir, bbox_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.bbox_dir = bbox_dir
        self.transform = transform

        # Read file names from the directories
        self.sketch_filenames = os.listdir(sketch_dir)

    def __len__(self):
        return len(self.sketch_filenames)

    def __getitem__(self, idx):
        # Read sketch and bounding box images
        sketch_filename = self.sketch_filenames[idx]

        sketch = Image.open(os.path.join(self.sketch_dir, sketch_filename)).convert('RGB')
        bbox = Image.open(os.path.join(self.bbox_dir, sketch_filename.replace('png', 'jpg'))).convert('RGB')

        # Apply transformations
        if self.transform is not None:
            sketch = self.transform(sketch)
            bbox = self.transform(bbox)

        return sketch, bbox, sketch_filename



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_resnet(*name_to_params('simclrv2/r152_3x_sk1.pth'))[0]
    model.fc = torch.nn.Identity()

    model = model.to(device)

        
    # Create the dataloader
    sketch_dir = "/DATA/nakul/sketch/data/validation/contours_seg"
    bbox_dir = "/DATA/nakul/sketch/data/validation/sr_bbox_regions"
    batch_size = 8

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = SketchDataset(sketch_dir, bbox_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=16)

    similarity_list = []

    # Iterate over the dataloader
    for batch in tqdm(dataloader):
        sketches, bboxes, sketch_filenames = batch

        sketch_features = model(sketches.to(device))
        bbox_features = model(bboxes.to(device))

        cosine_similarity = F.cosine_similarity(sketch_features, bbox_features).detach().cpu().tolist()
        similarity_list.extend(list(zip(cosine_similarity, sketch_filenames)))
    
    similarity_list.sort(reverse=True, key=lambda x: x[0])
    similarity_list = [x[-1] for x in similarity_list][:3600]

    with open('out/top_sketch_list.txt', 'w') as f:
        for x in similarity_list:
            f.write(f"{x}\n")
