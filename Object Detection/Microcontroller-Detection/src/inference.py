import numpy as np
import cv2
import torch
import glob as glob

from model import create_model
from utils import draw_boxes
from config import PREDICTION_THRES, NUM_EPOCHS, DEVICE

# load the model and the trained weights
model = create_model().to(DEVICE)
model.load_state_dict(torch.load(
    f'../output/model{NUM_EPOCHS}.pth', map_location=DEVICE
))
model.eval()

# directory where all the images are present
DIR_TEST = '../test_data'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")


# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = PREDICTION_THRES

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float32).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        labels = outputs[0]['labels'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        # draw the bounding boxes and write the class name on top of it
        orig_image = draw_boxes(boxes, labels, orig_image)
        cv2.imwrite(f"../test_prediction/{image_name}.jpg", orig_image)
        cv2.waitKey(0)
    print(f"Image {i+1} done...")
    print('-'*50)
print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()