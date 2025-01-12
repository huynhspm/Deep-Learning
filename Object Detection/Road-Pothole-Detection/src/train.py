import os
import matplotlib.pyplot as plt
import time

from utils import show_transformed_image
from engine import train, save_model
from config import VISUALIZE_TRANSFORMED_IMAGES, NUM_EPOCHS 
from config import SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH, OUT_DIR
from dataset import train_dataloader

if VISUALIZE_TRANSFORMED_IMAGES:
    show_transformed_image()

os.makedirs(OUT_DIR, exist_ok=True)
for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    start = time.time()
    train_loss_list, train_loss = train(train_dataloader)
    print(f"Epoch #{epoch} train loss: {train_loss:.3f}")   
    end = time.time()
    print(f"Took {(end - start) / 60} minutes for epoch {epoch}")
    
    figure, train_ax = plt.subplots()

    if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
        save_model(epoch + 1)
        print('SAVING MODEL COMPLETE...\n')
    
    if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
        train_ax.plot(train_loss_list, color='blue')
        train_ax.set_xlabel('iterations')
        train_ax.set_ylabel('train loss')
        figure.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
        print('SAVING PLOTS COMPLETE...')
    
    if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
        train_ax.plot(train_loss, color='blue')
        train_ax.set_xlabel('iterations')
        train_ax.set_ylabel('train loss')
        figure.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
        save_model(epoch + 1)
    
    plt.close('all')