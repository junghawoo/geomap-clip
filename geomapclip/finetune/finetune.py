import os
import torch
import torch.nn as nn
import re 
import pandas as pd
import sys
import logging

from torch.utils.data import DataLoader
from PIL import Image
from geomapclip.model.GeoMapCLIP import GeoMapCLIP
from geomapclip.model.misc import load_gps_data, file_dir
from geomapclip.train.train import train  # Import the train function
from geomapclip.train.eval import eval_images  # Import the eval_images function

from geomapclip.train.dataloader import GeoDataLoader, img_train_transform, img_val_transform  # Import GeoDataLoader and img_train_transform

import time  # Import the time module


logger = logging.getLogger(__name__)

def main():
    start_time = time.time()  # Start timing
    
    logging.basicConfig(
        filename="train.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Define transformations
    train_transform = img_train_transform()  # Use img_train_transform from dataloader.py
    val_transform = img_val_transform()  # Use img_val_transform from dataloader.py


    # osm terrain
    # zoom level 9 tiles 
    dataset_folder = "/home/ubuntu/images/tiles/terrain/osm/"
    # validation data may come from other provider
    val_dataset_folder = "/home/ubuntu/images/tiles/terrain/osm/" 


    train_dataset_file = "./train_longlat_10_zxy.csv"
    val_dataset_file = "./val_longlat_10_zxy.csv"
    
   
    batch_size = 256
    train_dataset = GeoDataLoader(train_dataset_file, dataset_folder, transform=train_transform)  # Use GeoDataLoader from dataloader.py
    print(f"Loaded train_dataset dataset with {len(train_dataset)} samples")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    print(f"Loaded train_dataloader dataset with {len(train_dataloader)} samples")

    
    val_dataset = GeoDataLoader(val_dataset_file, val_dataset_folder, transform=val_transform)  # Use GeoDataLoader from dataloader.py
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



    # Initialize model, optimizer, and other components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mini test =8
    # full = 4096
    model = GeoMapCLIP().to(device)

    #load our z9z10 gps gallery
    #model.gps_gallery = load_gps_data("./csv/coordinates_land_z9_z10.csv")
    model.gps_gallery = load_gps_data("./csv/fibonacci_lattice_land.csv")


    #load pretrained weights by us (GeoMapCLIP)
    # resuming from last epoch result
    # saved weights could be either cpu or gpu. Whatever it is, load the same device type as model
    #my_weights_folder = "/home/ubuntu/outputs/mix/04022025_ep15/weights"
    my_weights_folder = "./weights"
    model.image_encoder.mlp.load_state_dict(torch.load(f"{my_weights_folder}/image_encoder_mlp_weights.pth", map_location=device))
    model.location_encoder.load_state_dict(torch.load(f"{my_weights_folder}/location_encoder_weights.pth", map_location=device))
    model.logit_scale = nn.Parameter(torch.load(f"{my_weights_folder}/logit_scale_weights.pth", map_location=device))


    # Train the model
    total_epochs = 10 # 10 # 10 #20 # 10 # 20

    # LR 1e-4 for 32, 8e-4 for 256
    # GeoCLIP used 3e-5 so we used the same learnign rate. 03/23/2025
    LearningRate = 3e-5 # 1e-5  #3e-5
    #step_size = 10 #10 # 1 #10
    #gamma = 0.87  #0.87  #0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= step_size, gamma=gamma)
    #
    #This is cosine decay. This lets us stay aggressive early, but fine-tune later
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)


    weights_folder = "./weights"  # Define the path to the weights folder

    # Create the weights folder if it does not exist
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)


    # for 200km 
    best_val_acc = 0



    logger.info(' ')
    logger.info('New fine-tuning started ')
    logger.info(f'train dataset: {train_dataset_file}, validation dataset: {val_dataset_file}')

    for epoch in range(total_epochs):
        logger.info(f'Epoch {epoch+1}/{total_epochs} started.')
        train(train_dataloader, model, optimizer, epoch, batch_size=batch_size, device=device, scheduler=scheduler)
        logger.info(f'Epoch {epoch+1}/{total_epochs} completed.')

        # do validation 
        accuracy_results = eval_images(val_dataloader, model, device=device)
        logger.info(f'Epoch: {epoch+1}/{total_epochs}, learning rate={LearningRate}, batch_size={batch_size}.')

        val_acc = accuracy_results['acc_100_km']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {best_val_acc} at epoch {epoch}")
            logger.info(f"Saving model at epoch {epoch} with validation accuracy: {best_val_acc}")    
            # Save the fine-tuned model
            # if this epoch's evalutaion loss is the best, save the model 
            # Save the ImageEncoder's weights separately


            #move model to CPU to save weights in a safe format
            model.cpu() 

            torch.save(model.image_encoder.mlp.state_dict(), f"{weights_folder}/image_encoder_mlp_weights.pth")
            torch.save(model.location_encoder.state_dict(), f"{weights_folder}/location_encoder_weights.pth")
            torch.save(model.logit_scale, f"{weights_folder}/logit_scale_weights.pth")

            #keep training model using gpu
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
   
        #TODO
        #reload saved image_encoder's mlp states when resuming or testing with best weights

   

    logger.info(f"Total running time: {time.time() - start_time} seconds")  # End timing and print the elapsed time



    ## validation 
    #img_val_transform
    


if __name__ == "__main__":
    logger.info("started")
    main()
