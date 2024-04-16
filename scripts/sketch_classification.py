import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import torchvision.models as models
import os
import tqdm



class SketchDataset(Dataset):
    def __init__(self, base_dir, annotation_file, transform=None):
        self.data_dir = os.path.join(base_dir, 'contours_seg')
        self.transform = transform
        self.filenames = os.listdir(self.data_dir)

        with open(os.path.join(base_dir, annotation_file)) as f:
            self.annotation = json.load(f)
    
    def __getitem__(self, index):
        filename = self.filenames[index]

        f, idx = filename.replace('_out', '').split('.')[0].split('_')
        lbl = self.annotation[str(int(f))][int(idx)]['category_id']

        img = Image.open(os.path.join(self.data_dir, filename)).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return {
            'img': img,
            'label': lbl,
            'filename': filename
        }
    
    def __len__(self):
        return len(self.filenames)
    

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load label mapping dictionary
    with open('/DATA/nakul/sketch/ImageNet2COCO/coco2imagenet.json', 'r') as f:
        label_map = json.load(f)

    label_map = {int(x): y for x, y in label_map.items()}

    # Load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True).to(device)

    # Put model in evaluation mode
    model.eval()

    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create dataset and dataloader
    dataset = SketchDataset('/DATA/nakul/sketch/data/train', 'bbox_annotations.json', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1536, num_workers=8, pin_memory=True)

    # Initialize list to hold successfully classified filenames
    success_filenames = []

    # Iterate over batches and classify images
    with torch.no_grad():
        for batch, data in enumerate(tqdm.tqdm(dataloader)):
            # Extract data from batch
            imgs = data['img'].to(device)
            labels = data['label']
            filenames = data['filename']
            # Forward pass through model
            logits = model(imgs)
            # Get predicted class indices
            pred_indices = logits.argmax(dim=1)
            # Map predicted indices to class labels using label mapping dictionary
            valid_classes = [label_map.get(x.item(), [None]) for x in labels]

            # Check if any predictions match the true labels
            for i, pred in enumerate(pred_indices):
                # print(labels[i])
                # print(pred.item())
                # print(valid_classes[i])
                if pred.item() in valid_classes[i]:
                    success_filenames.append(filenames[i])
            
            # print(label_map)
            # exit()
        
                
    # Save list of successfully classified filenames to file
    with open('correctly_classified_sketches.txt', 'w') as f:
        for filename in success_filenames:
            f.write(f'{filename}\n')
