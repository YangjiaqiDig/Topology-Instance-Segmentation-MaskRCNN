from dataset import CoNSePDataset
import torch
import torch.utils.data
import numpy as np

def _coco_remove_images_without_annotations(dataset):
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

    assert isinstance(dataset, torch.utils.data.Dataset)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        _, anno, _ = dataset._CoNSePDataset__get_patch_item(ds_idx)
        if _has_valid_annotation(anno):
            ids.append(ds_idx)
    print('has_valid_annotation number of patches:', len(ids), '/', len(dataset.ids))
    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def get_consep(root, transforms, image_set='train'):
    dataset = CoNSePDataset(root, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset

