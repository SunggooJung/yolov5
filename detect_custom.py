from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
import torch
import numpy as np

import cv2


if __name__ == "__main__":
    # hyperparams
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False  # class-agnostic NMS
    max_det = 1  # maximum detections per image
    line_thickness = 3  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences

    # setup
    device = select_device("")
    weights = "runs/train/exp1/weights/best.pt"
    dnn = False  # use OpenCV DNN for ONNX inference
    half = False  # use FP16 half-precision inference
    data = "data/stair.yaml"  # dataset.yaml path
    imgsz = (640, 640)  # inference size (height, width)
    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load image
    path = "../datasets/stair/images/test/2022_1_10.jpg"
    im0 = cv2.imread(path)  # BGR
    img_size = 640
    stride = 32
    auto = True
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    # Run inference
    bs = 1  # batchsize
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    im = torch.from_numpy(im.copy()).to(device)
    im = torch.unsqueeze(im, 0)  # make a minibatch
    im = im.float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    augment = False
    visualize = False
    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # postprocess
    det = pred[0]
    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
    *xyxy, conf, cls = det[0]  # bbox pixels, confidence, class
    xmin, ymin, xmax, ymax = xyxy

    # Annotation
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    c = int(cls)
    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
    annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()

    cv2.imshow('image', im0)
    cv2.waitKey(0)
