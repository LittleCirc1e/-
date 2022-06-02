import os
import sys
import cv2
import time
import random
import shutil
from PIL import Image

import numpy as np
from os.path import join
import argparse
import subprocess
from tqdm import tqdm
import dlib
import multiprocessing
import json
import torch
import torch.nn
from torch.utils import data
from torchvision import transforms as trans

from xdistance import folder_select
from models import MAT_v2



#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
#gpu_ids = [*range(osenvs)]




class Worker():
    frame_size = 160
    frame_num = 20

    def __init__(self):

        self.model = self.load_model()
        self.tmp_path = "./tmp"

    def load_model(self):
        map_location = torch.device('cpu')
        ck = torch.load(
            "logs/saved_ck/7_auc_0.971792355371901_racc_0.865909090909091_facc_0.9472727272727273.pth", map_location='cpu')
        new_ck = {
            key[7:]: ck[key] for key in ck.keys()
        }

        model = MAT_v2(
            back_bone='xception', num_class=1, pretrained='imagenet', input_size=(self.frame_size, self.frame_size),
            attention_layers=["b7","b9"], atte_module_chocie="normal", M=8,
            feature_layer='b1', feature_choice='v2', DCT_choice=2,
            loss_choice='v2', margins=(0.5, [0.1, -2]), alpha=0.05,
            final_dim=256, cb_mode='fuse', rg_dropout_rate=0.25, fn_dropout_rate=0.5
        )
        model.load_state_dict(new_ck)
        model = model.to("cpu")
        model.eval()
        return model

    def work(self, path):
        if not os.path.exists(path + 'result.pth'):
            self.process_data(path)
            dataset = self.read_dataset('selected', self.frame_num)
            result = self.evaluate(dataset)
            stats = self.analyze(result)
            torch.save((stats, result), path + 'result.pth')
        else:
            stats, result = torch.load(path + 'result.pth')
        return (stats, result)

    def extract_frames(self, data_path):
        test_full_image_network(data_path, os.path.join(self.tmp_path, "frames"), 0, 100)

    def select_frame(self):
        folder_select(
            os.path.join(self.tmp_path, "frames"),
            os.path.join(self.tmp_path, "selected"),
            7)

    def process_data(self, path):
        self.refresh_dir()
        self.extract_frames(path)
        self.select_frame()

    def read_dataset(self, mode, frame_num):
        dataset = FFDataset(os.path.join(self.tmp_path, mode), frame_num, self.frame_size, augment=True)
        return dataset

    def refresh_dir(self):
        if os.path.exists(self.tmp_path):
            shutil.rmtree(self.tmp_path)
        os.mkdir(self.tmp_path)
        os.mkdir(os.path.join(self.tmp_path, "frames"))
        os.mkdir(os.path.join(self.tmp_path, "selected"))


    def evaluate(self, dataset):
        with torch.no_grad():
            y_pred = []
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=32,
                shuffle=False,
                num_workers=8
            )
            for img in dataloader:
                img = img.detach().cpu()
                output = self.model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())

        y_pred = np.array(y_pred)
        print(y_pred)

        return y_pred

    def analyze(self, y_pred):
        total_num = y_pred.shape[0]
        indices = np.where(y_pred > 0.5)[0]
        fake_num = len(indices)
        print(f"{fake_num}/{total_num}")
        if float(fake_num) / total_num > 0.5:
            return 1  # 1 for rue
        else:
            return 0


class FFDataset(data.Dataset):

    def __init__(self, dataset_root, frame_num=300, size=299, augment=True):
        self.data_root = dataset_root
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)
        if augment:
            self.transform = trans.Compose([trans.RandomHorizontalFlip(p=0.5), trans.ToTensor()])
            print("Augment True!")
        else:
            self.transform = trans.ToTensor()
        self.max_val = 1.
        self.min_val = -1.
        self.size = size

    def collect_image(self, root):
        image_path_list = []
        img_list = os.listdir(root)
        random.shuffle(img_list)
        img_list = img_list if len(img_list) < self.frame_num else img_list[:self.frame_num]
        for img in img_list:
            img_path = os.path.join(root, img)
            image_path_list.append(img_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size):
        img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        image_path = self.train_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img,size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        return len(self.train_list)

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def test_full_image_network(video_path, output_path,
                            start_frame=0, end_frame=None):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    #print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Face detector
    face_detector = dlib.get_frontal_face_detector()

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]
            # ------------------------------------------------------------------
            cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)), cropped_face)

        if frame_num >= end_frame:
            break

    pbar.close()





if __name__ == "__main__":
    detector = Worker()
    # print(
    #     detector.work("/home/ubuntu/qiufeng/dx/original_sequences/youtube/c23/videos/005.mp4")
    # )
    print(
        detector.work("C:/Users/20373/Desktop/dct/demo3/006_002.mp4")
    )