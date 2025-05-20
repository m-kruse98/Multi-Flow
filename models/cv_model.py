import torch
from torch import nn
from .cv_couplings import (
    NaiveCrossConvolutions,
    WrappedCrossConvolutions,
    NeighboringCrossConvolutions,
    SeparatedConvolutions,
    RandomNeighboringCrossConvolutions,
    TopNeighboringCrossConvolutions,
    NeighboringOnlyCrossConvolutions,
    NeighboringCrossAttention,
    NeighboringSelfAttention,
    parallel_glow_coupling_layer,
    ParallelPermute,
)
from .freia_funcs import (
    InputNode,
    Node,
    OutputNode,
    ReversibleGraphNet
)


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)

def concat_maps(maps):
    flat_maps = list()
    for m in maps:
        flat_maps.append(flat(m))
    return torch.cat(flat_maps, dim=1)[..., None] 

def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)


def get_cs_flow_model(config):
    input_dim = config["n_feat"]
    map_len = config["map_len"]
    nodes = list()
    
    if config["use_noise"]:
        nodes.append(InputNode(1, map_len, map_len, name="input0"))
    
    nodes.append(InputNode(input_dim, map_len, map_len, name='input1'))
    nodes.append(InputNode(input_dim, map_len, map_len, name='input2'))
    nodes.append(InputNode(input_dim, map_len, map_len, name='input3'))
    nodes.append(InputNode(input_dim, map_len, map_len, name='input4'))
    nodes.append(InputNode(input_dim, map_len, map_len, name='input5'))
    
    # choose right cs-flow model
    cross_convolution = {
        "cs_naive" : NaiveCrossConvolutions,
        "cs_wrapped" : WrappedCrossConvolutions,
        "cs_neigh" : NeighboringCrossConvolutions, 
        "cs_seperated" : SeparatedConvolutions,
        "cs_neigh_random" : RandomNeighboringCrossConvolutions,
        "cs_top" : TopNeighboringCrossConvolutions,
        "cs_neigh_only" : NeighboringOnlyCrossConvolutions,
        "cs_att_cross" : NeighboringCrossAttention,
        "cs_att_self" : NeighboringSelfAttention,
    }[config["arch"]]
    
    for k in range(config["n_coupling_blocks"]):
        if k == 0:
            node_to_permute = [nodes[-5].out0, nodes[-4].out0, nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
        else:
            node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2, nodes[-1].out3, nodes[-1].out4]

        nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
        input_list = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2, nodes[-1].out3, nodes[-1].out4]
        if config["use_noise"]:
            input_list.append(nodes[0].out0)
            
        nodes.append(Node(input_list,
                          parallel_glow_coupling_layer,
                          {'clamp': config["clamp"], 'F_class': cross_convolution, "use_noise" : config["use_noise"],
                           'F_args': {'channels_hidden': config["channels_hidden_teacher"],
                                      'kernel_size': config["kernel_sizes"][k], 'block_no': k}},
                          name=F'fc1_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
    nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
    nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
    nodes.append(OutputNode([nodes[-4].out3], name='output_end2'))
    nodes.append(OutputNode([nodes[-5].out4], name='output_end2'))
    nf = ReversibleGraphNet(nodes, n_jac=5)
    return nf


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        if not config["pre_extracted"]:
            raise NotImplementedError("Please pre-extract using the preprocess.py!")

        self.use_noise = config["use_noise"]        
        self.net = get_cs_flow_model(config)
        self.config = config
    
    
    def loss(self, z, jac, per_sample=False, per_pixel=False, mask=None, means=0, n_views=5):
        """ Compute the loss by reordering the samples

        Args:
            z (list(torch.Tensor)): list with output in latent space of the model of shape (B, C, H, W) for each of the views
            jac (torch.Tensor): jacobian tensor
            per_sample (bool, optional): per-sample anomaly score/loss. Defaults to False.
            per_pixel (bool, optional): Whether to return a per-pixel anomaly score/loss. Defaults to False.
            mask (torch.Tensor, optional): Mask to only grab features from the foreground of the object. Defaults to None.
            means (int, optional): _description_. Defaults to 0.

        Returns:
            torch.Tensor: loss value per pixel (shape [N_views * B, H, W]) or per sample (shape [N_views * B,])
        """
        
        # Each entries in the list z contains the output for one view of the objects in the batch, and they 
        #                  A                     B       ...        E
        # comes in as: [B, 304, 24,24], [B, 304, 24,24], ..., [B, 304, 24, 24]
        # per_pixel we need: [5B, 24, 24]
        # per_sample we need: [5B,]
        # with the 5B being structured as [A1,B1,C1,D1,E1, A2,B2, ...., AB,BB,CB,DB,EB]
        
        B = z[0].shape[0]
        z = torch.cat(z, dim=0)
        idx = torch.arange(z.shape[0])
        result = (idx % n_views) * B + (idx // n_views)
        
        pixel_scores = (0.5 * torch.sum((mask.unsqueeze(1) * z[result,...] - means) ** 2, dim=1) - mask * jac[result, ...]) / z.shape[1]
        
        if per_pixel:
            return pixel_scores
        elif per_sample:
            return pixel_scores.mean(dim=(-1,-2))
        return pixel_scores.mean()
    
    def forward(self, x):
        if not self.config["pre_extracted"]: #  and c.mode != 'depth':
            with torch.no_grad():
                f = self.feature_extractor(x)
        else:
            f = x

        if self.use_noise:
            b_size = x[0].shape[0]            
            
            if self.use_noise == 1:
                # simplenet noise conditioning
                c = (torch.tensor([0.15 for _ in range(b_size)]) if self.training else torch.zeros(b_size))[...,None, None, None]
                noise = torch.randn((b_size, 1, self.config["map_len"], self.config["map_len"])) * c
            elif self.use_noise == 2:
                # softflow noise conditioning
                c = (torch.rand(b_size) if self.training else torch.zeros(b_size))[...,None, None, None]
                noise = torch.randn((b_size, 1, self.config["map_len"], self.config["map_len"])) * (c * c)
            elif self.use_noise == 3:
                # uniform noise conditioning
                noise = torch.rand((b_size, 1, self.config["map_len"], self.config["map_len"]))
                if not self.training:
                    noise *= 0
            
            noise = noise.to(self.config["device"])
            f = [x_entry + noise for x_entry in x]
            f = [noise * 20, *f]
            
        inp = f
        z = self.net(inp)        
        
        jac = torch.cat(self.net.jacobian(run_forward=False), dim=0)
        return z, jac

