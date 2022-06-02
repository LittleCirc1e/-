import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm
import dlib
import multiprocessing
import json

DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']

train = []
valid = []
test = []


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


def extract_frames(data_path, output_path, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    if os.path.exists(output_path):
        return
    os.makedirs(output_path, exist_ok=True)
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        '''
        reader = cv2.VideoCapture(data_path)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                        image)
            frame_num += 1
        reader.release()
        '''
        test_full_image_network(data_path, output_path)
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))



def extract_method_videos(ran, data_path, dataset, compression):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    #os.makedirs(images_path, exist_ok=True)
    files = list(range(ran * 100, (ran + 1) * 100))
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        num = image_folder.split('_')[0]
        if dataset == 'original':
            if int(str(image_folder)) not in files:
                continue
            elif image_folder in train:
                images_path = join("./ffpp/train", DATASET_PATHS[dataset], compression, 'images')
            elif image_folder in valid:
                images_path = join("./ffpp/valid", DATASET_PATHS[dataset], compression, 'images')
            else:
                images_path = join("./ffpp/test", DATASET_PATHS[dataset], compression, 'images')
        else:
            if int(str(num)) not in files:
                continue
            elif num in train:
                images_path = join("./ffpp/train", DATASET_PATHS[dataset], compression, 'images')
            elif num in valid:
                images_path = join("./ffpp/valid", DATASET_PATHS[dataset], compression, 'images')
            else:
                images_path = join("./ffpp/test", DATASET_PATHS[dataset], compression, 'images')
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder))


if __name__ == '__main__':
    with open("splits/train.json", "r") as f:
        train_dict = json.load(f)
        train = sum(train_dict, [])
        f.close()
    with open("splits/val.json", "r") as f1:
        valid_dict = json.load(f1)
        valid = sum(valid_dict, [])
        f1.close()
    with open("splits/test.json", "r") as f2:
        test_dict = json.load(f2)
        test = sum(test_dict, [])
        f2.close()
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c0')
    args = p.parse_args()

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(ran, **vars(args))
    else:
        processes = []
        for ran in range(10):
            processes.append(multiprocessing.Process(target = extract_method_videos, args = (ran, args.data_path, args.dataset, args.compression, )))
            processes[-1].start()
        for process in processes:
            process.join()
