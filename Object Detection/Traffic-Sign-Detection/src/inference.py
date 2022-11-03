import numpy as np
import cv2
import os
import torch
import glob
import time

from config import(
    DEVICE, NUM_EPOCHS, RESIZE_TO, BACKBONE, 
    INFERENCE_DIR, TEST_DIR, OUT_DIR, 
    PREDICTION_THRES, NUM_CLASSES, MIN_SIZE 
)
from model import create_model
from custom_utils import draw_boxes

if __name__ == '__main__':
    # load the best model and trained weights
    backbone = BACKBONE
    epoch = NUM_EPOCHS
    model = create_model(NUM_CLASSES, backbone, MIN_SIZE)
    model_dir = f"{OUT_DIR}/train/{backbone}_model_{epoch}.pth"
    checkpoint = torch.load(model_dir, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_images = []
    if os.path.isdir(TEST_DIR):
        test_images = glob.glob(f"{TEST_DIR}/*.ppm")
    else:
        test_images.append(TEST_DIR)
    print(f"Test instances: {len(test_images)}")


    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = PREDICTION_THRES

    # to count the total number of frames iterated through
    frame_count = 0
    # to keep adding the frames' FPS
    total_fps = 0
    # number of image to predict
    NUMBER_IMAGES = 5
    for i in range(NUMBER_IMAGES):
        # get the image file name for saving output later on
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # resize bounding
        image = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
        
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float32).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        end_time = time.time()

        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
        boxes = outputs[0]['boxes'].data.numpy()
        labels = outputs[0]['labels'].data.numpy()            
        scores = outputs[0]['scores'].data.numpy()

        # filter out boxes according to `detection_threshold`      
        boxes = boxes[scores >= detection_threshold].astype(np.int32)

        print(f"{len(boxes)} object in image")
        # resize bounding boxes to fit with original image
        boxes[:, 0] = (boxes[:, 0]/RESIZE_TO)*orig_image.shape[1]
        boxes[:, 1] = (boxes[:, 1]/RESIZE_TO)*orig_image.shape[0]
        boxes[:, 2] = (boxes[:, 2]/RESIZE_TO)*orig_image.shape[1]
        boxes[:, 3] = (boxes[:, 3]/RESIZE_TO)*orig_image.shape[0]

        # draw the bounding boxes and write the class name on top of it
        orig_image = draw_boxes(boxes, labels, orig_image)
                
        # save imag
        cv2.imwrite(f"{INFERENCE_DIR}/image/{image_name}.jpg", orig_image)

        print(f"{image_name} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")