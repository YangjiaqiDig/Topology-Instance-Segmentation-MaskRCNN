import os
import numpy as np
import torch
import scipy.io
from PIL import Image
import transforms as T

'''
'inst_map' is a 1000x1000 array containing a unique integer for each individual nucleus. i.e the map ranges from 0 to N, 
    where 0 is the background and N is the number of nuclei.
'type_map' is a 1000x1000 array where each pixel value denotes the class of that pixel. 
    The map ranges from 0 to 7, where 7 is the total number of classes in CoNSeP.       
'inst_type' is a Nx1 array, indicating the type of each instance (in order of inst_map ID)       
'inst_centroid' is a Nx2 array, giving the x and y coordinates of the centroids of each instance (in order of inst map ID).
'''


class CoNSePDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'Images'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'Labels'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        mask_path = os.path.join(self.root, "Labels", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask_mat = scipy.io.loadmat(mask_path)
        mask = mask_mat['inst_map']
        # convert the Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)  # === len(mask_mat['inst_type'])+1 -> N + 1
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]  # N
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]  # (N, 1000, 1000) True/False

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.tensor(mask_mat['inst_type'], dtype=torch.int64).squeeze()  # torch[N]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        print('hehhe')
        transforms.append(T.RandomIoUCrop())
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':
    root = 'data/CoNSeP/train'
    dataset = CoNSePDataset(root, get_transform(train=True))
    print(dataset[0][0].shape)
    test = np.unique([2,1,6, 4, 8])
    print(test)
