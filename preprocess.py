import os
from tqdm import tqdm
import copy
import argparse

import torch
import ttach as tta
from models.extractor import FeatureExtractor
from models.MVANet import inf_MVANet
from torch.autograd import Variable

import cv2
import numpy as np
from utils import t2np, build_dataloader
from config import dataset
from scipy.ndimage import binary_fill_holes, binary_dilation


def extract_image_features(classname_to_do, extract_layer=35):
    """For extracting with efficientnet

    Args:
        base_dir (_type_): _description_
        extract_layer (_type_, optional): _description_. Defaults to c.extract_layer.
    """
    model = FeatureExtractor(layer_idx=extract_layer)
    model.to("cuda")
    model.eval()
    config = {copy.copy(a):copy.copy(v) for (a,v) in dataset.items()}
    
    train_meta = config["train"]["meta_file"]
    test_meta = config["train"]["meta_file"]
    config["input_size"] = (768, 768)
    
    for class_name in [classname_to_do]:
        
        config["batch_size"] = 32
        config["train"]["meta_file"] = f"{train_meta}/{class_name}.json"
        config["test"]["meta_file"] = f"{test_meta}/{class_name}.json"
        config["classname"] = class_name
        config["type"] = "explicit"
        train_loader, test_loader = build_dataloader(config, distributed=True)
        
        model.to("cuda")
        for name, loader in zip(['train', 'test'], [train_loader, test_loader]):
            features = list()
            for i, data in enumerate(tqdm(loader)):
                # data has fields: filename, height, width, label, clsname, maskname, image, mask
                img = data["image"].to("cuda")
                
                with torch.no_grad():
                    z = model(img)
                features.append(t2np(z))

            features = np.concatenate(features, axis=0)
            export_dir = os.path.join(config["feature_dir"], class_name)
            os.makedirs(export_dir, exist_ok=True)
            print(f"Saving features of shape {features.shape} to {export_dir}")
            np.save(os.path.join(export_dir, f'{name}.npy'), features)


def extract_background(class_name):
        
    config = {copy.copy(a):copy.copy(v) for (a,v) in dataset.items()}
    train_meta = config["train"]["meta_file"]
    test_meta = config["train"]["meta_file"]
    config["input_size"] = (1024,1024)
    
    net = inf_MVANet().cuda()
    pretrained_dict = torch.load(os.path.join("models/", 'Model_80.pth'), map_location='cuda')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.eval()
    forward_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        ]
    )
    
    config["batch_size"] = 32
    config["train"]["meta_file"] = f"{train_meta}/{class_name}.json"
    config["test"]["meta_file"] = f"{test_meta}/{class_name}.json"
    config["classname"] = class_name
    config["type"] = "explicit"
    print(config)
    train_loader, test_loader = build_dataloader(config, distributed=True)

    for dataloader, is_train in zip([train_loader, test_loader], [True, False]):
        masks = list()
        for b_idx, batch in enumerate(dataloader):
            img = batch["image"].to("cuda")
            
            for idx in tqdm(range(img.shape[0])):
                cur_img = img[idx]
                with torch.no_grad():
                    img_var = Variable(cur_img.unsqueeze(0)).cuda().float()
                    mask = []
                    for transformer in forward_transforms:
                        rgb_trans = transformer.augment_image(img_var)
                        model_output = net(rgb_trans)
                        deaug_mask = transformer.deaugment_mask(model_output)
                        mask.append(deaug_mask)

                prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid().round().int().squeeze().cpu().numpy()
                prediction = binary_fill_holes(prediction)
                prediction = binary_dilation(prediction, structure=np.ones((8,8)))
                prediction = cv2.resize(prediction.astype(np.uint8), dsize=(256,256), interpolation=cv2.INTER_NEAREST)
                masks.append(prediction[None])
            
        masks = np.concatenate(masks, axis=0)
        export_dir = os.path.join("tmp", "masks", class_name)
        os.makedirs(export_dir, exist_ok=True)
        print(f"Saving masks of shape {masks.shape} to {export_dir}")
        np.save(os.path.join(export_dir, f'{"train" if is_train else "test"}.npy'), masks)
        


parser = argparse.ArgumentParser()
parser.add_argument("-c", "-classname", metavar="c", type=str,
                    default="pcb")
args, extras = parser.parse_known_args()

extract_image_features(args.c)
extract_background(args.c)
