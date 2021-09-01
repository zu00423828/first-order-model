import face_alignment
import skimage.io
import numpy
from argparse import ArgumentParser
from tqdm import tqdm
import os
# import imageio
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")
def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = cv2.resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    print(xA,yA,xB,yB)
    print(interArea,boxAArea,boxBArea)
    return iou
iou=bb_intersection_over_union([0,0,10,10],[5,5,15,15])
print(iou)
# video=cv2.VideoCapture("dst.mp4")
# while video.isOpened():
#     ret,frame=video.read()
#     if not ret:
#         break

