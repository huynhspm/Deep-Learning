import torch

BACKBONE = 'resnet50' # mobilenet_v3_large  ;  mobilenet_v3_large_320
BATCH_SIZE = 2
# RESIZE_TO = 512
NUM_EPOCHS = 5
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = '../road-pothole-images-for-pothole-detection/Dataset 1 (Simplex)/Dataset 1 (Simplex)/Train data'
TEST_DIR = '../road-pothole-images-for-pothole-detection/Dataset 1 (Simplex)/Dataset 1 (Simplex)/Test data'

CLASSES = ['__background__', 'pothole']
NUM_CLASSES = 2
LEARNING_RATE = 0.001

MIN_SIZE = 800
VISUALIZE_TRANSFORMED_IMAGES = False # whether to visualize images after crearing the data loaders
PREDICTION_THRES = 0.8
OUT_DIR = '../output'
SAVE_PLOTS_EPOCH = 1 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 1 # save model after these many epochs