import torch

# determine the backbone for Faster RCNN network
BACKBONE = 'resnet50' 
# mobilenet_v3_large  -  mobilenet_v3_large_320

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 5 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '../microcontroller-detection/Microcontroller Detection/train'
# validation images and XML files directory
VALID_DIR = '../microcontroller-detection/Microcontroller Detection/test'
# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
]
NUM_CLASSES = 5
MIN_SIZE = 800
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
PREDICTION_THRES = 0.8
# location to save model and plots
OUT_DIR = '../output'
SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs