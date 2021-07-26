import os, sys
import numpy as np
import torch
import scipy.io
from PIL import Image
import transforms as T

np.set_printoptions(threshold=sys.maxsize)

'''
'inst_map' is a 1000x1000 array containing a unique integer for each individual nucleus. i.e the map ranges from 0 to N, 
    where 0 is the background and N is the number of nuclei.
'type_map' is a 1000x1000 array where each pixel value denotes the class of that pixel. 
    The map ranges from 0 to 7, where 7 is the total number of classes in CoNSeP.       
'inst_type' is a Nx1 array, indicating the type of each instance (in order of inst_map ID)       
'inst_centroid' is a Nx2 array, giving the x and y coordinates of the centroids of each instance (in order of inst map ID).
'''
def get_accurate_mask(mask_mat, with_type=True):
    ann_inst = mask_mat['inst_map']
    inst_type = mask_mat['inst_type']
    if with_type:
        ann_type = mask_mat["type_map"]
        # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
        # If own dataset is used, then the below may need to be modified
        ann_type[(ann_type == 3) | (ann_type == 4)] = 3
        ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4
        inst_type[(inst_type == 3) | (inst_type == 4)] = 3
        inst_type[(inst_type == 5) | (inst_type == 6) | (inst_type == 7)] = 4

        ann = np.dstack([ann_inst, ann_type])
        ann = ann.astype("int32")
    else:
        ann = np.expand_dims(ann_inst, -1)
        ann = ann.astype("int32")
    return ann, inst_type


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


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
        ann, inst_type = get_accurate_mask(mask_mat)
        # convert the Image into a numpy array
        mask = np.array(ann[..., 0])
        type_map = ann[..., 1].copy()
        # instances are encoded as different colors
        obj_ids = np.unique(mask)  # === len(mask_mat['inst_type'])+1 -> N + 1
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]  # N
        # masks = mask == obj_ids[:, None, None]  # (N, 1000, 1000) True/False

        # get bounding box coordinates for each mask
        boxes = []
        masks = []
        labels = []
        for i in obj_ids:
            mask_map = mask == i
            pos = np.where(mask_map)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax == xmin or ymax == ymin:
                type_map[mask_map] = 0
                continue
            # inst_map = np.array(mask == i + 1, np.uint8)
            # print('hover', get_bounding_box(inst_map))
            boxes.append([xmin, ymin, xmax, ymax])
            masks.append(mask_map)
            labels.append(inst_type.squeeze()[i - 1])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.tensor(labels, dtype=torch.int64)  # torch[N], 4 class + background
        masks = np.stack(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# def get_transform(train):
#     transforms_list = []
#     transforms_list.append(T.ToTensor())
#     if train:
#         print('inside')
#         transforms_list.append(T.RandomHorizontalFlip(0.5))
#         print('after one')
#         transforms_list.append(T.RandomIoUCrop())
#         print('done')
#         # transforms.append(T.RandomHorizontalFlip(0.5))
#     transforms_list.append(T.ToTensor())
#     print('last compose')
#     return T.Compose(transforms_list)


def get_transform(train):
    return T.DetectionPresetTrain() if train else T.DetectionPresetEval()


if __name__ == '__main__':
    root = 'data/CoNSeP/train'
    dataset = CoNSePDataset(root, get_transform(train=True))
    for each in dataset:
        var = each
    test = np.unique([2, 1, 6, 4, 8])
    print(test)
