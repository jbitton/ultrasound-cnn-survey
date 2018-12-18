# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from maskrcnn_benchmark.data.datasets.nerve_detection import NERVEDetection

class NERVEDataset(NERVEDetection):
    CLASSES = (
        "__background__ ",
        "nerve",
    )

    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(NERVEDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            new_ids = []
            for img_id in self.ids:
                anns = self.nerve.loadAnns(self.nerve.getAnnIds(imgIds=img_id, iscrowd=None))
                is_filled = True
                for ann in anns:
                    if not ann["bbox"] or not ann["segmentation"]:
                        is_filled = False
                if is_filled and len(anns) > 0:
                    new_ids.append(img_id)
            self.ids = new_ids
            #self.ids = [
            #    img_id
            #    for img_id in self.ids
            #    if len(self.nerve.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            #]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.nerve.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        import sys
        sys.stdout.flush()
        print('getting item', idx)
        img, anno = super(NERVEDataset, self).__getitem__(idx)
        # filter crowd annotations
        # TODO might be better to add an extra field
        # anno = [obj for obj in anno if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        # was remove_empty=True
        target = target.clip_to_image(remove_empty=False)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.nerve.imgs[img_id]
        return img_data

