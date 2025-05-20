import numpy as np
import torch
from PIL import ImageFile
import heapq
from datasets.data_builder import build_dataloader
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import wandb

def train_dataset(train_function, config):
    
    if config["wandb"]:
        wandb.init(project=config["project"], config={c:a for c,a in config.items() if c != "data_config"},
                   name=config["prefix"], mode="online", settings=wandb.Settings(start_method='thread'))
        wandb.define_metric("train_loss", step_metric="train_step")
        wandb.define_metric("test_loss", step_metric="test_step")
        wandb.define_metric("NF_samplewise_mean", step_metric="epoch")
        wandb.define_metric("NF_samplewise_max", step_metric="epoch")
        wandb.define_metric("NF_mean_image_roc", step_metric="epoch")
        wandb.define_metric("NF_pixel_roc", step_metric="epoch")
        wandb.define_metric("NF_aupro", step_metric="epoch")
        wandb.define_metric("NF_max_image_roc", step_metric="epoch")
                
    data_config = config["data_config"]
    
    train_loader, test_loader = build_dataloader(data_config, distributed=False)
    train_function(train_loader, test_loader, config=config)



class AnomalyTracker:
    """
    A class for tracking the top N anomalies and normal samples based on their anomaly scores.

    Attributes:
        top_n (int): The number of top anomalies and normal samples to track.
        anomalies (list): A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top anomalies.
        normals (list): A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top normal samples.
    """

    def __init__(self, top_n=100):
        """
        Initializes the tracker with a specified top_n value.

        Args:
            top_n (int, optional): The number of top anomalies and normal samples to track. Defaults to 100.
        """
        self.top_n = top_n
        self.anomalies = []  # (anomaly_score, filename, anomaly_map, gt_mask, image)
        self.normals = []  # (anomaly_score, filename, anomaly_map, gt_mask, image)

    def update(self, anomaly_score, filename, anomaly_map, gt_mask, label, image):
        """
        Updates the tracker with a new sample.

        If the sample is an anomaly (label=1), it is added to the anomalies list.
        If it is a normal sample (label=0), it is added to the normals list.

        Args:
            anomaly_score (float): The anomaly score of the sample.
            filename (str): The filename of the sample.
            anomaly_map (object): The anomaly map of the sample.
            gt_mask (object): The ground truth mask of the sample.
            label (int): The label of the sample (0 for normal, 1 for anomaly).
            image (np.ndarray): Image of the sample
        """
        if label == 1:  # anomaly
            if len(self.anomalies) < self.top_n:
                heapq.heappush(self.anomalies, (-anomaly_score, filename, anomaly_map, gt_mask, image))
            else:
                heapq.heappushpop(self.anomalies, (-anomaly_score, filename, anomaly_map, gt_mask, image))
        else:  # normal
            if len(self.normals) < self.top_n:
                heapq.heappush(self.normals, (anomaly_score, filename, anomaly_map, gt_mask, image))
            else:
                heapq.heappushpop(self.normals, (anomaly_score, filename, anomaly_map, gt_mask, image))

    def get_top_anomalies(self):
        """
        Returns the top N anomalies, sorted in descending order of their anomaly scores.

        Returns:
            list: A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top anomalies.
        """
        return sorted(self.anomalies, reverse=True)

    def get_top_normals(self):
        """
        Returns the top N normal samples, sorted in descending order of their anomaly scores.

        Returns:
            list: A list of tuples containing the anomaly score, filename, anomaly map, and ground truth mask for the top normal samples.
        """
        return sorted(self.normals, reverse=True)
    

def get_instancewise_data(data, config):
    labels, image, features = data["label"], data["image"], data["feature"]
    features = to_device([features], config["device"])[0]
    mask = data["mask"]
    img_in = features if config["pre_extracted"] else image
    cameras = data["camera"]
    
    return img_in, labels, image, mask, cameras, data["filename"]

def get_samplewise_data(data, config):
    
    # here B is the batch_size
    B = data["feature_0"].shape[0]
    idx = torch.arange(B * 5)
    result = (idx % 5) * B + (idx // 5)
    
    
    labels = torch.cat(data["label"])[result]
    
    images = torch.cat([data["image_0"],data["image_1"],data["image_2"],data["image_3"],data["image_4"]], dim=0)[result,...]
    features = to_device([data["feature_0"], data["feature_1"], data["feature_2"], data["feature_3"], data["feature_4"]], config["device"])
    masks = torch.cat([data["mask_0"],data["mask_1"],data["mask_2"],data["mask_3"],data["mask_4"]], dim=0)[result,...]
    if config["rem_bg"]:
        foregrounds = torch.cat([data["foreground_0"],data["foreground_1"],data["foreground_2"],data["foreground_3"],data["foreground_4"]], dim=0)[result,...].to("cuda")
    else:
        B, C, H, W = features[0].shape
        foregrounds = torch.ones((5 * B, H, W)).to("cuda")
    
    filenames = np.concatenate(data["filename"])[result]
    cameras = torch.cat(data["cameras"])[result]
    
    return features, labels, images, masks, cameras, filenames, foregrounds

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def to_device(tensors, device):
    return [t.to(device) for t in tensors]


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name, percentage=True):
        self.name = name
        self.max_epoch = 0
        self.best_score = None
        self.last_score = None
        self.percentage = percentage

    def update(self, score, epoch, print_score=False):
        if self.percentage:
            score = score * 100
        self.last_score = score
        improved = False
        if epoch == 0 or score > self.best_score:
            self.best_score = score
            improved = True
        if print_score:
            self.print_score()
        return improved

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t best: {:.2f}'.format(self.name, self.last_score, self.best_score))


def model_size_info(model):
    # Get the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Get the size of the model in MB
    model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 * 1024)
    
    # Format the output string
    output = f"**Model Size Info**\n"
    output += f"  * Number of Parameters: {num_params:,}\n"
    output += f"  * Model Size (MB): {model_size_mb:.2f} MB"
    
    return output


def save_weights(model, class_name, suffix, device="cuda"):
    save_to = "checkpoints"
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    model.to('cpu')
    torch.save(model.net.state_dict(), os.path.join(save_to, f'{class_name}_{suffix}.pth'))
    print('model saved')
    model.to(device)


def load_weights(model, class_name, suffix, device="cuda"):
    print("loading:", os.path.join("checkpoints", f'{class_name}_{suffix}.pth'))
    model.net.load_state_dict(torch.load(os.path.join("checkpoints", f'{class_name}_{suffix}.pth')))
    model.eval()
    model.to(device)
    return model