from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transforms
from glob import glob
from os import path
import numpy as np
import xml.etree.ElementTree as ET


IMAGE_PATH = 'data/images'
ANNOTATION_PATH = 'data/annotations'
SHAPE_X = 304
SHAPE_Y = 304

TRANSFORM = transforms.Compose(
    [
    transforms.ColorJitter(0.5, 0.8, 0.8, 0.5),
    transforms.Resize((SHAPE_X, SHAPE_Y)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.3, 0.3, 0.3], [0.2, 0.2, 0.2]),
    transforms.ToHeatmap((SHAPE_X, SHAPE_Y))
    ]
)

BASE_TRANSFORM = transforms.Compose(
        [
        transforms.Resize((SHAPE_X, SHAPE_Y)),
        transforms.ToTensor()
        ]
)

class RoadSignData(Dataset):
    def __init__(self, image_path, annotation_path, transform = TRANSFORM, base_transform = BASE_TRANSFORM):
        """Dataset class to load in images and annotations"""
        # Load images
        self.image_data = []
        for f in glob(path.join(image_path, '*.png')):
            i = Image.open(f).convert('RGB')
            i.load()
            self.image_data.append(i)
        # Load annoation boxes
        self.annotation_data = []
        for i, f in enumerate(glob(path.join(annotation_path, '*.xml'))):
            # Pull in the image and label things
            image_size = self.image_data[i].size
            root = ET.parse(f).getroot()
            names = root.findall("./object/name")
            xmins = root.findall("./object/bndbox/xmin")
            ymins = root.findall("./object/bndbox/ymin")
            xmaxes = root.findall("./object/bndbox/xmax")
            ymaxes = root.findall("./object/bndbox/ymax")
            # Process each label component
            this_file_annotation = []
            for name, xmin, ymin, xmax, ymax in zip(names, xmins, ymins, xmaxes, ymaxes):
                name = name.text
                # Encode objects
                if name == 'trafficlight':
                    object = 0
                elif name == 'stop':
                    object = 1
                elif name =='speedlimit':
                    object = 2
                elif name =='crosswalk':
                    object = 3
                else:
                    object = 4
                # Pull bounding box points
                xmin = int(xmin.text)
                ymin = int(ymin.text)
                xmax = int(xmax.text)
                ymax = int(ymax.text)
                this_file_annotation.append([object, xmin, ymin, xmax, ymax])
            self.annotation_data.append(this_file_annotation)
        self.transform = transform
        self.base_transform = base_transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_data = self.image_data[idx]
        location_data = self.annotation_data[idx]
        image_data, location_data = self.transform(image_data, location_data)
        unprocessed_image = self.image_data[idx]
        unprocessed_image = self.base_transform(unprocessed_image)
        return image_data, location_data, unprocessed_image


def load_data(image_path = IMAGE_PATH, annotation_path = ANNOTATION_PATH, transform = TRANSFORM, num_workers=0, batch_size=32):
    dataset = RoadSignData(image_path, annotation_path, transform = transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == '__main__':
    # Plot example of photo and label
    import matplotlib.pyplot as plt
    data = load_data()
    for transformed_image, heatmap, unprocessed_image in data:
        fig, axs = plt.subplots(8, 7)
        for i in range(8):
            img0 = transformed_image[i].permute(1, 2, 0)
            axs[i, 0].imshow(img0)
            tl0 = heatmap[i][0]
            axs[i, 1].imshow(tl0, cmap='gray')
            s0 = heatmap[i][1]
            axs[i, 2].imshow(s0, cmap='gray')
            sl0 = heatmap[i][2]
            axs[i, 3].imshow(sl0, cmap='gray')
            cw0 = heatmap[i][3]
            axs[i, 4].imshow(cw0, cmap='gray')
            e0 = heatmap[i][4]
            axs[i, 5].imshow(e0, cmap='gray')
            img0r = unprocessed_image[0][i].permute(1, 2, 0)
            axs[i, 6].imshow(img0r)
        plt.show()
        break