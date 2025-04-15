import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from geopy.distance import geodesic as GD

logger = logging.getLogger(__name__)

def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    gd_avg = 0

    for i in range(total):
        gd = GD(gps_gallery[preds[i]], targets[i]).km
        gd_avg += gd
        if gd <= dis:
            correct += 1

    gd_avg /= total
    return correct / total, gd_avg

def distance_list(targets, preds, gps_gallery=None):
    total = len(targets)
    errors =[]
    
    for i in range(total):
        gd = GD(gps_gallery[preds[i]], targets[i]).km
        errors.append(gd)

    return errors

def eval_images(val_dataloader, model, device="cpu"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):


            imgs = torch.from_numpy(np.asarray(imgs))
            # Stack along dimension 0 (creates shape [2, N])
            labels_tensor = torch.stack(labels)
            # Transpose to get shape (N, 2)
            labels_tensor = labels_tensor.T  # or gps_tensor.transpose(0, 1)



            #print("before", imgs.dtype, labels_tensor.dtype)
            imgs = imgs.float()
            labels = labels_tensor.float()
            #print("after", imgs.dtype, labels_tensor.dtype)

            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)
            
            # Predict gps location with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            
            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()


    # add 100, 40 km
    distance_thresholds = [2500, 750, 200, 100, 40, 25, 1] # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        logger.info(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")

        accuracy_results[f'acc_{dis}_km'] = acc

    return accuracy_results



def eval_images_verbose(val_dataloader, model, device="cpu"):
    """
    This function is variant of eval_image that returns 
    the list of individual prediction and target. 
    Returning individual prediction and its error helps anlyzing which tiles have the biggest errors.
    
    """
    model.eval()
    preds = []
    targets = []
    errors =[]
    
    gps_gallery = model.gps_gallery

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):


            imgs = torch.from_numpy(np.asarray(imgs))
            # Stack along dimension 0 (creates shape [2, N])
            labels_tensor = torch.stack(labels)
            # Transpose to get shape (N, 2)
            labels_tensor = labels_tensor.T  # or gps_tensor.transpose(0, 1)



            #print("before", imgs.dtype, labels_tensor.dtype)
            imgs = imgs.float()
            labels = labels_tensor.float()
            #print("after", imgs.dtype, labels_tensor.dtype)

            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)
            
            # Predict gps location with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            # print("outs:", outs)
            # print("probs:", probs)
            
            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    errors = distance_list(targets, preds, gps_gallery) 
    #print("errors in eval_images_verbose:")
    #print(errors)
    model.train()


    # add 100, 40 km
    distance_thresholds = [2500, 750, 200, 100, 40, 25, 1] # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        logger.info(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")

        accuracy_results[f'acc_{dis}_km'] = acc

    return preds, targets, errors, accuracy_results
