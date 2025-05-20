from __future__ import division
from typing import List, Dict, Union

import json
import logging
import os.path as osp

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
from pathlib import Path
import re
import cv2
logger = logging.getLogger("global_logger")


def extract_camera_number(filename):
    match = re.search(r'C(\d)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("A weird file has been found?")

def build_feature_dataloader(cfg, training, distributed=False):

    image_reader = build_image_reader(cfg["image_reader"])

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building ExplicitDataset from: {}".format(cfg["meta_file"]))

    dataset = FeatureDataset(
        image_reader,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
        feature_dir=cfg["feature_dir"],
        classname=cfg["classname"],
        samplewise=cfg["samplewise"],
        rem_bg=cfg["rem_bg"]
    )

    if not training:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True,
        drop_last=training
    )

    return data_loader


# TODO: maybe make loading the image optional, because it may save time (and memory)
class FeatureDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        meta_file,
        training,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
        feature_dir=None,
        classname=None,
        samplewise=False,
        rem_bg=True
    ):
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn
        self.samplewise = samplewise
        self.rem_bg = rem_bg
        if isinstance(self.meta_file, str):
            self.meta_file = [meta_file]

        # construct metas
        self.metas = sum((self.load_explicit(path, self.training)
                          for path in self.meta_file), [])
        self.feature_dir = feature_dir
        self.classname = classname
        
        self.features = np.load(Path(feature_dir, self.classname, f"{'train' if self.training else 'test'}.npy"))
        
        if self.rem_bg:
            self.foreground = np.load(Path("tmp", "masks", self.classname, f"{'train' if self.training else 'test'}.npy"))
            self.foreground = np.stack([cv2.resize(mask, (24, 24), cv2.INTER_NEAREST) for mask in self.foreground])
            print(f"Loaded foreground masks of shake {self.foreground.shape}:")
   
        
    @staticmethod
    def load_explicit(path: str, is_training: bool) -> List[Dict[str, Union[str, int]]]:
        SAMPLE_KEYS = {'category', 'anomaly_class', 'image_path', 'mask_path'}

        with open(path, 'r') as fp:
            info = json.load(fp)
            assert isinstance(info, dict) and all(
                key in info for key in ('meta', 'train', 'test')
            )
            meta = info['meta']
            train = info['train']
            test = info['test']
            raw_samples = train if is_training else test

        assert isinstance(raw_samples, list) and all(
            isinstance(sample, dict) and set(sample.keys()) == SAMPLE_KEYS
            for sample in raw_samples
        )
        assert isinstance(meta, dict)
        prefix = meta['prefix']
        normal_class = meta['normal_class']

        if is_training:
            return [dict(filename=osp.join(prefix, sample['image_path']),
                         label_name=normal_class, label=0,
                         clsname=sample['category'])
                    for sample in raw_samples]
        else:
            def as_normal(sample):
                return (sample['mask_path'] is None or
                        sample['anomaly_class'] == normal_class)

            return [dict(
                filename=osp.join(prefix, sample['image_path']),
                maskname=None if as_normal(sample)
                else osp.join(prefix, sample['mask_path']),
                label=0 if as_normal(sample) else 1,
                label_name=sample['anomaly_class'],
                clsname=sample['category']
            ) for sample in raw_samples]

    def __len__(self):
        return len(self.metas) if not self.samplewise else (len(self.metas) // 5)

    def __getitem__(self, index):
        return self.get_instancewise(index) if not self.samplewise else self.get_samplewise(index)
    
    
    def get_instancewise(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]
        label = meta["label"]
        image = self.image_reader(meta["filename"])
        feature = self.features[index]
        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
                "feature" : feature,
                "foregound" : self.foreground[index] if self.rem_bg else 0,
                "camera" : extract_camera_number(filename),
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]

        image = Image.fromarray(image, "RGB")

        # read / generate mask
        if meta.get("maskname", None):
            input['maskname'] = meta['maskname']
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            input['maskname'] = ''
            if label == 0:  # good
                mask = np.zeros((image.height, image.width)).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        input.update({"image": image, "mask": mask})
        return input
    
    
    def get_samplewise(self, index):
        input = {}
        start_idx = index * 5
        
        cur_metas = [self.metas[idx] for idx in range(start_idx, start_idx + 5)]

        labels = [c["label"] for c in cur_metas]
        filenames = [a["filename"] for a in cur_metas]
        input.update(
            {
                "filename": filenames,
                # TODO: image loading is currently very inefficient. 256,256 probably suffices
                "height": 768,
                "width": 768,
                "label": labels,
                "feature_0" : self.features[start_idx],
                "feature_1" : self.features[start_idx + 1],
                "feature_2" : self.features[start_idx + 2],
                "feature_3" : self.features[start_idx + 3],
                "feature_4" : self.features[start_idx + 4],
                "cameras" : [extract_camera_number(f) for f in filenames],
            }
        )
        
        if self.rem_bg:
            input.update(
                {
                "foreground_0" : self.foreground[start_idx],
                "foreground_1" : self.foreground[start_idx + 1],
                "foreground_2" : self.foreground[start_idx + 2],
                "foreground_3" : self.foreground[start_idx + 3],
                "foreground_4" : self.foreground[start_idx + 4],
                }
            )

        # TODO: re-work the CLSname into the loading, because it may be beneficial for visualization/logging
        # if meta.get("clsname", None):
        #     input["clsname"] = meta["clsname"]
        # else:
        #     input["clsname"] = filename.split("/")[-4]

        # TODO: also grab the images at later times!
        
        # read / generate mask
        for meta_idx, meta in enumerate(cur_metas):
            
            image = self.image_reader(meta["filename"])
            image = Image.fromarray(image, "RGB")
            
            if meta.get("maskname", None):
                input['maskname'] = meta['maskname']
                mask = self.image_reader(meta["maskname"], is_mask=True)
            else:
                input['maskname'] = ''
                if labels[meta_idx] == 0:  # good
                    mask = np.zeros((image.height, image.width)).astype(np.uint8)
                elif labels[meta_idx] == 1:  # defective
                    mask = (np.ones((image.height, image.width)) * 255).astype(np.uint8)
                else:
                    raise ValueError("Labels must be [None, 0, 1]!")
        
            mask = Image.fromarray(mask, "L")

            if self.transform_fn:
                image, mask = self.transform_fn(image, mask)
            if self.colorjitter_fn:
                image = self.colorjitter_fn(image)
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            if self.normalize_fn:
                image = self.normalize_fn(image)
            input.update({f"image_{meta_idx}": image, f"mask_{meta_idx}": mask})
        return input
