import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse
import wandb
from models.cv_model import Model
import config

from utils import (
    AnomalyTracker,
    Score_Observer,
    model_size_info,
    t2np,
    train_dataset,
    get_instancewise_data,
    get_samplewise_data,
    save_weights
)

from viz import (
    compare_histogram,
    visualize
)

from adeval import EvalAccumulatorCuda

def train(train_loader, test_loader, config):
    samplewise = config["data_config"]["samplewise"] == 1
    model = Model(config=config)
    model.to(config["device"])
        
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=config["lr"], eps=1e-08,
                                  weight_decay=1e-5, betas=(0.9,0.95))
    print(model_size_info(model))
    
    get_data = get_samplewise_data if samplewise else get_instancewise_data
    
    
    mean_nll_obs = Score_Observer('AUROC mean over maps')
    max_nll_obs = Score_Observer('AUROC  max over maps')
    samplewise_mean_nll_obs = Score_Observer('Sample-wise AUROC mean over maps')
    samplewise_max_nll_obs = Score_Observer('Sample-wise AUROC  max over maps')
    pixel_st_obs = Score_Observer('AUROC pixel-wise')
    aupro_st_obs = Score_Observer('AUPRO for segmentation')
    
    # NOTE: Comment this in to track the worst & best performing samples
    # failure_tracker = AnomalyTracker(top_n=100)
    train_iter, test_iter = 0,0
    train_clamp = (torch.inf, -torch.inf)
    
    
    for epoch in range(config["meta_epochs"]):
        # train some epochs
        model.train()
        if config["verbose"]:
            print(F'\nTrain epoch {epoch}')
        for sub_epoch in tqdm(range(config["sub_epochs"])):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=config["hide_tqdm_bar"])):
                optimizer.zero_grad()
                
                img_in, labels, image, mask, cameras, filenames, foregrounds = get_data(data, config)
                
                z, jac = model(img_in)
                loss = model.loss(z,jac,mask=foregrounds)
                
                if config["wandb"]:
                    wandb.log({"train_loss" : loss.item(), "train_step" : train_iter})
                    train_iter += 1
                train_loss.append(t2np(loss))
                
                cat = torch.cat(z).detach().cpu()
                train_clamp = (min(train_clamp[0], torch.amin(cat).item()), max(train_clamp[1], torch.amax(cat).item()))
                
                loss.backward()
                optimizer.step()

            mean_train_loss = np.mean(train_loss)
            if config["verbose"] and sub_epoch % 4 == 0:  # and epoch == 0:
                print('Epoch: {:d}.{:d} \t train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

        accum = EvalAccumulatorCuda(train_clamp[0], train_clamp[1], train_clamp[0], train_clamp[1],
                                    nstrips=10000)        
        print("TRAIN CLAMPS:", train_clamp)
        # evaluate
        model.eval()
        if config["verbose"]:
            print('\nCompute loss and scores on test set:')
        test_loss = list()
        test_labels = list()
        img_nll = list()
        max_nlls = list()        
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=config["hide_tqdm_bar"])):
                
                img_in, labels, image, mask, cameras, filenames, foregrounds = get_data(data, config)                
                
                z, jac = model(img_in)
                
                loss = model.loss(z, jac, mask=foregrounds, per_sample=True)
                nll = model.loss(z, jac, mask=foregrounds, per_pixel=True)

        
                
                if nll.amin() < train_clamp[0]:
                    print(i, "Warning: Clamping outside of min", nll.amin())
                if nll.amax() > train_clamp[1]:
                    print(i, "Warning: Clamping outside of max", nll.amax())
                # prepare anomaly map & masks for evaluation
                ano_map = torch.nn.functional.interpolate(nll.unsqueeze(1), (256,256), mode="bilinear")
                ano_map = torch.clamp(ano_map.squeeze(), train_clamp[0], train_clamp[1]).cuda(non_blocking=True)
                mask = mask.to(torch.uint8).squeeze().cuda(non_blocking=True)
                img_score = torch.amax(nll,dim=(-1, -2))
                # efficient accumulated calculation of AUROC/AUPRO 
                accum.add_anomap_batch(ano_map, mask)
                accum.add_image(torch.clamp(img_score, train_clamp[0], train_clamp[1]), labels)
                
                # NOTE: Comment this in to track the worst & best performing samples
                # if epoch == config["meta_epochs"] - 1:
                #     for idx in range(mask.shape[0]):
                #         failure_tracker.update(
                #             anomaly_score=img_score[idx],
                #             filename=filenames[idx],
                #             anomaly_map=t2np(ano_map[idx]),
                #             gt_mask=t2np(mask[idx]),
                #             label=labels[idx].item(),
                #             image=image[idx].numpy()
                #             )
                        
                img_nll.append(t2np(loss))
                max_nlls.append(np.max(t2np(nll), axis=(-1, -2)))
                test_loss.append(loss.mean().item())
                if config["wandb"]:
                    wandb.log({"test_loss" : test_loss[-1], "test_step" : test_iter})
                    test_iter += 1
                test_labels.append(labels)
        
        
        img_nll = np.concatenate(img_nll)
        max_nlls = np.concatenate(max_nlls)
        test_loss = np.mean(np.array(test_loss))    
        
        if config["verbose"]:
            print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))
        
        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        sample_wise_labels = test_labels.reshape((-1, 5)).any(axis=1)
        instance_wise_scores_max = max_nlls.reshape((-1, 5)).mean(axis=1)
        instance_wise_scores_mean = img_nll.reshape((-1, 5)).mean(axis=1)
        
        for lbl, score in zip(sample_wise_labels, instance_wise_scores_max):
            accum.add_sample(score, lbl)
        
        print(accum.summary())
        
        metrics = accum.summary()
        if epoch == config["meta_epochs"] - 1: 
            compare_histogram(img_nll, test_labels, config["class_name"], config["prefix"], name=f"imagewise_mean", thresh=5)
            compare_histogram(max_nlls, test_labels, config["class_name"], config["prefix"], name="imagewise_max", thresh=5)
            compare_histogram(instance_wise_scores_mean, sample_wise_labels, config["class_name"],
                              config["prefix"], name="samplewise_mean", thresh=5, n_bins=64)
            compare_histogram(instance_wise_scores_max, sample_wise_labels, config["class_name"],
                              config["prefix"], name="samplewise_max", thresh=5, n_bins=64)
        
        
        mean_nll_obs.update(roc_auc_score(is_anomaly, img_nll), epoch,
                            print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        max_nll_obs.update(roc_auc_score(is_anomaly, max_nlls), epoch, # metrics["i_auroc"]
                           print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        samplewise_mean_nll_obs.update(roc_auc_score(sample_wise_labels, instance_wise_scores_mean), epoch,
                            print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
        samplewise_max_nll_obs.update(roc_auc_score(sample_wise_labels, instance_wise_scores_max), epoch, # 
                           print_score=config["verbose"] or epoch == config["meta_epochs"] - 1)
    
        aupro_st_obs.update(metrics["p_aupro"], epoch, True)
        pixel_st_obs.update(metrics["p_auroc"], epoch, True)
        
        
        if config["wandb"]:
            wandb.log({
                "epoch" : epoch,
                "NF_samplewise_mean" : samplewise_mean_nll_obs.last_score,
                "NF_samplewise_max" : samplewise_max_nll_obs.last_score,
                "NF_mean_image_roc" : mean_nll_obs.last_score,
                "NF_pixel_roc" : pixel_st_obs.last_score,
                "NF_aupro" : aupro_st_obs.last_score,
                "NF_max_image_roc" : max_nll_obs.last_score
            })
        accum.reset()
        
    # NOTE: Comment this in to track the worst & best performing samples
    # norms, anos = failure_tracker.get_top_normals(), failure_tracker.get_top_anomalies()
    # visualize(norms, config["prefix"], config["class_name"], 0, train_clamp[1] / 8, is_ano=False)
    # visualize(anos, config["prefix"], config["class_name"], 0, train_clamp[1] / 8, is_ano=True)
    
    if config["save_model"]:
        save_weights(model, config["class_name"], config["prefix"], config["device"])

    return mean_nll_obs, max_nll_obs, pixel_st_obs, aupro_st_obs


if __name__ == "__main__":
    
    arch_choices = [
        "cs_naive", # csflow with naive cross-view connections
        "cs_wrapped", # csflow with cross-view along camera_idx
        "cs_neigh", # csflow with cross-view on neighboring views
        "cs_seperated", # seperated convolutions for each view. essentially AST teacher fitted into one network
        "cs_neigh_fixed",
        "cs_neigh_random",
        "cs_top",
        "cs_neigh_only",
        "cs_att_cross",
        "cs_att_self",
    ]
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", metavar="p", type=str,
                        default="to_delete")
    parser.add_argument("-project", type=str, default="03_csflow_realiad")
    parser.add_argument("-seed", type=int, default=10000)
    parser.add_argument("-wandb", type=int, default=0)
    parser.add_argument("-save_model", type=int, default=0)
    parser.add_argument("-use_noise", type=int, default=1)
    parser.add_argument("-rem_bg", type=int, default=1)
    parser.add_argument("-c", "-classname", metavar="c", type=str,
                        default="pcb")
    parser.add_argument("-multi", metavar="b", type=int, help="0 for regular training. 1 for multi-class",
                        default=0)
    parser.add_argument("-arch",  type=str, help="Chose type of architecture", choices=arch_choices,
                        default="cs_neigh")
    parser.add_argument("-samplewise", metavar="b", type=int,
                        help="0 for mixing the training images freely. 1 for sorting them by instance/sample",
                        default=1)
    args, extras = parser.parse_known_args()
    
    print(args)
    assert args.samplewise if "cs" in args.arch else not args.samplewise, f"For CS-Flow architecture {args.arch}, samplewise dataloading is needed!"
    
    class_name = args.c    
    config_obj = config.effnet_config
    config_obj["prefix"] = args.prefix
    config_obj["project"] = args.project
    config_obj["wandb"] = args.wandb == 1
    config_obj["arch"] = args.arch
    config_obj["class_name"] = class_name
    config_obj["multi"] = args.multi
    config_obj["seed"] = args.seed
    config_obj["rem_bg"] = args.rem_bg == 1
    config_obj["data_config"]["rem_bg"] = args.rem_bg == 1
    config_obj["use_noise"] = args.use_noise
    config_obj["data_config"]["samplewise"] = args.samplewise
    config_obj["save_model"] = args.save_model == 1
    
    torch.manual_seed(args.seed)
    config_obj["data_config"]["train"]["meta_file"] = f"{config_obj['data_config']['train']['meta_file']}/{class_name}.json"
    config_obj["data_config"]["test"]["meta_file"] = f"{config_obj['data_config']['test']['meta_file']}/{class_name}.json"
    config_obj["data_config"]["classname"] = class_name
        
    
    if args.samplewise:
        # since samplewise results in a batch_size *= 5, we reduce it a bit
        config_obj["data_config"]["batch_size"] = 8
    else:
        raise NotImplementedError("This code & data-loading has to be executed sample-wise to work properly.")
    
    print(f"Executing for classname: {class_name}")
    train_dataset(train, config=config_obj)
