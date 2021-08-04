import os, sys
from matplotlib.pyplot import box
import numpy as np
import torch
import scipy.io as sio
from PIL import Image, ImageDraw
import transforms as T
from convert_image import *
from torchvision.transforms import functional as F
import glob
import cv2
from consep_utils import *
np.set_printoptions(threshold=sys.maxsize)


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


####
class __Kumar(__AbstractDataset):
    """Defines the Kumar dataset as originally introduced in:

    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, 
    and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for 
    computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CPM17(__AbstractDataset):
    """Defines the CPM 2017 dataset as originally introduced in:

    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban, 
    Talha Qaiser, Navid Alemi Koohbanani et al. "Methods for segmentation and classification 
    of digital microscopy tissue images." Frontiers in bioengineering and biotechnology 7 (2019).

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CoNSeP(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in 
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563
    
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann


####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "kumar": lambda: __Kumar(),
        "cpm17": lambda: __CPM17(),
        "consep": lambda: __CoNSeP(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name

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
    if with_type:
        ann_type = mask_mat["type_map"]
        # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
        # If own dataset is used, then the below may need to be modified
        ann_type[(ann_type == 3) | (ann_type == 4)] = 3
        ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

        ann = np.dstack([ann_inst, ann_type])
        ann = ann.astype("int32")
    else:
        ann = np.expand_dims(ann_inst, -1)
        ann = ann.astype("int32")
    return ann

class CoNSePDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.ids = []
        data_dir = os.path.join(self.root, '250x250_50x50')
        self.ids.extend(glob.glob("%s/*.npy" % data_dir))
        self.ids.sort()  # to always ensure same input ordering

    def __get_patch_item(self, idx):
        path = self.ids[idx]
        data = np.load(path)

        # split stacked channel into image and label
        imgArray = (data[..., :3]).astype("uint8")  # RGB images [H, W, C]
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map
        img = Image.fromarray(imgArray, 'RGB')
        return img, ann[..., 0], ann[..., 1]

        # if self.shape_augs is not None:
        #     shape_augs = self.shape_augs.to_deterministic()
        #     img = shape_augs.augment_image(img)
        #     ann = shape_augs.augment_image(ann)

    def __getitem__(self, idx):
        img, mask, type_map = self.__get_patch_item(idx)
        # instances are encoded as different colors, first id is the background, so remove it
        # masks = mask == obj_ids[:, None, None]  # (N, 1000, 1000) True/False
        obj_ids = np.unique(mask)[1:]  # === len(mask_mat['inst_type']) -> N 
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
            boxes.append([xmin, ymin, xmax, ymax]) # coordinators
            masks.append(mask_map) # True / False for each instance, append N instance True/False map
            label = type_map[mask_map]
            labels.append(np.unique(label)[0]) # There are whole image number of instances' labels(classes), pick the one on certain instance

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)  # torch[N], 4 class + background
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((1), dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)
        target = {
            "boxes": boxes, # coordinators
            "labels": labels,   # class numbers
            "masks": masks,     # True/False maps
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        if self.transforms is not None:
            if len(boxes) > 0:
                img, target = self.transforms(img, target)
            else:
                img = F.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform(train):
    return T.DetectionPresetTrain() if train else T.DetectionPresetEval()


if __name__ == '__main__':
    root = 'data/training_data/consep/valid'
    dataset = CoNSePDataset(root, get_transform(train=True))
    # dataset[800]
    _, anno, _ = dataset._CoNSePDataset__get_patch_item(800)
    def _has_only_empty_bbox(anno):
        obj_ids = np.unique(anno)[1:]
        num_valid_ids = []
        for i in obj_ids:
            mask_map = anno == i
            pos = np.where(mask_map)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmax == xmin or ymax == ymin:
                continue 
            num_valid_ids.append(i)
        return len(num_valid_ids) == 0
    def _has_only_background(anno):
        return anno.sum() == 0

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        if _has_only_background(anno):
            return False
        return True
    if _has_valid_annotation(anno):
        print('innnnn')
    ss
    for i in range(27):
        # print(i)
        dataset[i]
    # test = np.unique([2, 1, 6, 4, 8])
    # print(test)
