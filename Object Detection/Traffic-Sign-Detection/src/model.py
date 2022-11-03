import torchvision.models.detection as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes, backbone, min_size):
    # load Faster RCNN pre-trained model
    if backbone == 'resnet50':
        model = models.fasterrcnn_resnet50_fpn(
            pretrained=True, min_size=min_size)
    elif backbone == 'mobilenet_v3_large':
        model = models.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=True, min_size=min_size)
    elif backbone == 'mobilenet_v3_large_320':
        model = models.fasterrcnn_mobilenet_v3_large_320_fpn(
            pretrained=True, min_size=min_size)
    else:
        print('error in backbone config')

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace pre-trained head with our features head
    # the head layer will classify the images based on our data input features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model