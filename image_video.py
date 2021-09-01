import imageio
import torch
import numpy as np
import cv2
from datetime import datetime
import skvideo.io 
def sv_read():
    st=datetime.now()
    video=skvideo.io.vread("0.mp4")
    # print(video.shape)
    et=datetime.now()
    print((et-st).total_seconds())
def image_read():
    st=datetime.now()
    video=[]
    reader=imageio.get_reader("0.mp4")
    for im in reader:
        video.append(im)
    et=datetime.now()
    print((et-st).total_seconds())
def cv2_read():
    st=datetime.now()
    video=[]
    reader=cv2.VideoCapture("0.mp4")
    while reader.isOpened():
        ret,frame=reader.read()
        if not ret:
            break
        video.append(frame)
    video_array=np.array(video)
    et=datetime.now()
# for i in range(10):
#     sv_read()
#     image_read()
#     cv2_read()
# reader.release()
# video_t=torch.tensor(np.array(video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
# print(video_t.shape)
# out_video=torch.ones(video_t.shape[2],video_t.shape[1],video_t.shape[3],video_t.shape[4])
# for idx in range(video_t.shape[2]):
#     out_video[idx]=video_t[:,:,idx]
# print(out_video.shape)
# out_video=out_video.permute(0,2,3,1)[:,:,:,[2,1,0]].type(torch.uint8).numpy()
# imageio.mimsave("test1.mp4", out_video, fps=30)
# print("video reder finish")
import os
from torch.utils.data import Dataset
import glob
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread
from sklearn.model_selection import train_test_split
def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = None
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            print("check")
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out
from torch.utils.data import DataLoader
if __name__=="__main__":
    dataset = FramesDataset( root_dir="demo/video/",is_train=True)
    # print(len(dataset))
    print(dataset[0]["name"])
    dl=DataLoader(dataset)
    for x in dl:
        print(x["name"])