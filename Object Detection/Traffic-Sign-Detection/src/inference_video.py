import numpy as np
import cv2
import torch
import os
import time
import argparse

from config import(
    DEVICE, NUM_EPOCHS, RESIZE_TO, 
    BACKBONE, OUT_DIR, INFERENCE_DIR,
    PREDICTION_THRES, NUM_CLASSES, MIN_SIZE 
)
from model import create_model
from custom_utils import draw_boxes

if __name__ == '__main__':
    # load the best model and trained weights
    backbone = 'resnet50'
    epoch = NUM_EPOCHS
    model = create_model(NUM_CLASSES, backbone, MIN_SIZE)
    model_dir = f"{OUT_DIR}/train/{backbone}_model_{epoch}.pth"
    checkpoint = torch.load(model_dir, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = PREDICTION_THRES

    name_video = "video_1_trimmed_1.mp4"
    cap = cv2.VideoCapture(f"../input//video/{name_video}")

    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # define codec and create VideoWriter object 
    out = cv2.VideoWriter(f"{INFERENCE_DIR}/video/{name_video}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))

    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second

    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, image = cap.read()
        if ret:
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
            image = torch.tensor(image, dtype=torch.float).cuda()
            # add batch dimension
            image = torch.unsqueeze(image, 0)
            # get the start time
            start_time = time.time()
            with torch.no_grad():
                # get predictions for the current frame
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

            # resize bounding boxes to fit with original image
            boxes[:, 0] = (boxes[:, 0]/RESIZE_TO)*orig_image.shape[1]
            boxes[:, 1] = (boxes[:, 1]/RESIZE_TO)*orig_image.shape[0]
            boxes[:, 2] = (boxes[:, 2]/RESIZE_TO)*orig_image.shape[1]
            boxes[:, 3] = (boxes[:, 3]/RESIZE_TO)*orig_image.shape[0]

            # draw the bounding boxes and write the class name on top of it
            orig_image = draw_boxes(boxes, labels, orig_image)
            out.write(orig_image)
            # press `q` to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()

    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")