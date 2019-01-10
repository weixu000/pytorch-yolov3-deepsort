import json
import os

from torch.utils.data import Dataset

from util import DurationTimer


class COCODataset(Dataset):
    """
    Represents COCO Dataset
    See http://cocodataset.org/#format-data for annotation format
    """

    def __init__(self, annotation_file, image_folder, transform=None):
        self.transform = transform if transform else lambda *x: x

        self.bbox, self.cls, self.imgs, = [], [], []
        self.bbox_cls, self.img_bbox = [], []

        print('loading annotations into memory...')
        with DurationTimer() as timer:
            self.dataset = json.load(open(annotation_file, 'r'))
            self.create_index(image_folder)
        print(f'Done (t={timer.duration:0.2f}s)')

    def create_index(self, image_folder):
        def process_item(item, transform):
            ind = {x['id']: i for i, x in enumerate(item)}  # Map id to index into the list
            item = [transform(x) for x in item]  # Transform items in self.dataset
            return ind, item

        ann_ind, self.bbox = process_item(self.dataset['annotations'], lambda x: x['bbox'])
        img_ind, self.imgs = process_item(self.dataset['images'], lambda x: os.path.join(image_folder, x['file_name']))
        cat_ind, self.cls = process_item(self.dataset['categories'], lambda x: x['name'])

        self.img_bbox = [[] for _ in range(len(self.imgs))]
        self.bbox_cls = [0 for _ in range(len(self.bbox))]
        for ann in self.dataset['annotations']:
            self.img_bbox[img_ind[ann['image_id']]].append(ann_ind[ann['id']])
            self.bbox_cls[ann_ind[ann['id']]] = cat_ind[ann['category_id']]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        bbox = [self.bbox[x] for x in self.img_bbox[idx]]  # [x, y, width, height]
        cls = [self.bbox_cls[x] for x in self.img_bbox[idx]]

        return self.transform(img, bbox, cls)
