import numpy as np
import cv2
import os
import torch

from tqdm import tqdm
from config import DEVICE, NUM_EPOCHS, TEST_DIR, PREDICTION_THRES
from model import create_model
from utils import draw_boxes

model = create_model().to(DEVICE)
model.load_state_dict(torch.load(
    f'../output/model{NUM_EPOCHS}.pth', map_location=DEVICE
))
model.eval()

test_images = os.listdir(TEST_DIR)
print(f"Validation instances: {len(test_images)}")


# initialize tqdm progress bar
prog_bar = tqdm(test_images, total=len(test_images))

cnt = 20
for i, image in enumerate(prog_bar):
    orig_image = cv2.imread(f'{TEST_DIR}/{test_images[i]}')
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # convert to tensor
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
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
        boxes = boxes[scores >= PREDICTION_THRES].astype(np.int32)

        # draw the bounding boxes and write the class name on top of it
        orig_image = draw_boxes(boxes, labels, orig_image)
        cv2.imwrite(f"../test_prediction/{test_images[i]}", orig_image)
        cv2.waitKey(0)
    print()
    print(f"Image {i+1} done...")
    print('-'*50)    
    cnt -= 1
    if cnt == 0: break
print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()