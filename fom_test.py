import face_alignment
import cv2
import os
import glob
import numpy as np
orgin_video=cv2.VideoCapture("dst.mp4")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                    face_detector="blazeface",device='cpu')
offset_size=360
def compute_aspect_preserved_bbox(bbox, increase_area):
    left, top, right, bot = bbox[0],bbox[1],bbox[2],bbox[3]
    width = right - left
    height = bot - top
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))
    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)
    return (left, top, right, bot)
def offset(left,top,right,bot):
    h_offset=(offset_size-(bot-top))/2
    w_offfset=(offset_size-(right-left))/2
    left=int(left-w_offfset)
    right=int(right+w_offfset)
    top=int(top-h_offset)
    bot=int(bot+h_offset)
    return left,top,right,bot
def extract_bbox(frame):
    bbox=fa.face_detector.detect_from_image(frame)
    bbox=bbox[0].astype(np.int16)
    left,top,right,bot=compute_aspect_preserved_bbox(bbox,0.1)
    left,top,right,bot=offset(left,top,right,bot)
    frame=frame[top:bot,left:right]
    frame=cv2.resize(frame,(256,256))
    return frame
def preprocess():
    fourcc=cv2.VideoWriter_fourcc(*'MP4V')
    out=cv2.VideoWriter('out.mp4',fourcc,30,(256,256))
    while orgin_video.isOpened():
        ret,frame=orgin_video.read()
        if not ret:
            break
        frame=extract_bbox(frame)
        out.write(frame)
    orgin_video.release()
    out.release()
preprocess()


