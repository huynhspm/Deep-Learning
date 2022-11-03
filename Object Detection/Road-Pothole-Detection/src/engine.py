import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.optim import SGD, Adam

from config import DEVICE, LEARNING_RATE, OUT_DIR
from model import create_model
from dataset import train_dataloader
from utils import Averager

model = create_model()
model = model.to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
MODEL_NAME = 'model'
train_loss_list = []
train_loss = Averager()

def train(train_loader):
    print('Training')

    # initialize tqdm progress bar
    prog_bar = tqdm(train_dataloader, total=len(train_dataloader))
    train_loss.reset()
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_list.append(loss_value)
        train_loss.send(loss_value)

        losses.backward()
        optimizer.step()

        # update the loss value beside the progress bar for each iteration
        if i % 25 == 0:
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_loss_list, train_loss.value
    
       
def save_model(id: int):
    torch.save(model.state_dict(), f'{OUT_DIR}/{MODEL_NAME}{id}.pth')








