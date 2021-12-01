import matplotlib
from numpy.lib import source
matplotlib.use('Agg')
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
import torch
from sync_batchnorm import DataParallelWithCallback
from skimage.transform import resize
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
import moviepy.editor as mp
import subprocess
import requests
import cv2

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()
        kp_detector.cuda()
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector







class Animation():
    def __init__(self,source,first_driving,generator,kp_detector,relative=True, adapt_movement_scale=True):
        self.source= torch.tensor(source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        first_driving=torch.tensor(np.array(first_driving)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        self.generator=generator
        self.kp_detector=kp_detector
        self.kp_source = self.kp_detector(self.source)
        self.kp_driving_initial=self.kp_detector(first_driving)
        self.relative=relative
        self.adapt_movement_scale=adapt_movement_scale
    def __call__(self, driving_frame):
        with torch.no_grad():
            driving_frame = torch.tensor(np.array(driving_frame)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)#b.c,h,w
            driving_frame = driving_frame#.cuda()
            # print('drive frame',driving_frame.shape)
            kp_driving = self.kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=self.kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=self.kp_driving_initial, use_relative_movement=self.relative,
                                    use_relative_jacobian=self.relative, adapt_movement_scale=self.adapt_movement_scale)
            out = self.generator(self.source, kp_source=self.kp_source, kp_driving=kp_norm)
        return (np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1]) )[0]

def img_color_resize(img):
    img=resize(img,(256,256))
    # return img[...,[2,1,0]]
    return img[...,:3]
def main_cv2():
    # source_image = cv2.imread(opt.source_image)
    # source_image=img_color_resize(source_image)
    source_image=laod_stylegan_avatar()
    video=cv2.VideoCapture(opt.driving_video)
    animation=None
    fps = video.get(5)
    frame_count = int(video.get(7))
    out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'FFV1'), fps, (256, 256))
    from tqdm import trange
    for idx in trange(frame_count):
        ret,driving_frame=video.read()
        driving_frame=img_color_resize(driving_frame)
        if animation  is None :
            animation=Animation(source_image,driving_frame,generator,kp_detector)
        predict=animation(driving_frame)
        predict=np.array(predict*255,dtype=np.uint8)
        out.write(predict)
    out.release()
    video.release()
    origin_video=mp.VideoFileClip(opt.driving_video)
    audio=origin_video.audio
    new_video=mp.VideoFileClip('out.avi')
    new_video.audio=audio
    new_video.write_videofile('finish.mp4')

def laod_stylegan_avatar():
    url= "https://thispersondoesnotexist.com/image"
    r = requests.get(url, headers={'User-Agent': "My User Agent 1.0"}).content
    image = np.frombuffer(r, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize(image, (256, 256))
    return image

def main_imageio():
    source_image = imageio.imread(opt.source_image)
    source_image=img_color_resize(source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    animation=None
    predict_list=[]
    print('process')
    try:
        for im in reader:
            driving_frame=img_color_resize(im)
            if animation is None:
                animation=Animation(source_image,driving_frame,generator,kp_detector)
            predict=animation(driving_frame)
            predict_list.append(predict)
    except RuntimeError:
        pass
    reader.close()
    print(len(predict_list))
    imageio.mimsave("temp.mp4", [(frame*255).astype(np.uint8) for frame in predict_list], fps=fps)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    print('start')
    main_cv2()
    # main_imageio()
    print('finish')
            


