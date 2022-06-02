import numpy as np
import os
import cv2

def compute_distance(pic1, pic2, dimension):
    return np.sqrt(np.sum(np.square(pic1 - pic2))/dimension)

def folder_select(folder_path, output_path, threshold):
    files = os.listdir(folder_path)
    num = 0
    frames = {}
    for i in range(len(files) - 1):
        pic1 = cv2.imread(os.path.join(folder_path, files[i]))
        pic2 = cv2.imread(os.path.join(folder_path, files[i + 1]))
        if pic1 is None or pic2 is None:
            continue
        (h1, w1, _) = pic1.shape
        (h2, w2, _) = pic2.shape
        '''
        if h1 > h2:
            pic1 = cv2.resize(pic1, (h2, h2), interpolation = cv2.INTER_CUBIC)
            pic2 = cv2.resize(pic2, (h2, h2), interpolation = cv2.INTER_CUBIC)
        else:
            pic1 = cv2.resize(pic1, (h1, h1), interpolation = cv2.INTER_CUBIC)
            pic2 = cv2.resize(pic2, (h1, h1), interpolation = cv2.INTER_CUBIC)
        '''
        if h1 != h2:
            continue
        dist = compute_distance(pic1, pic2, h1*h1*3)
        if dist > threshold:
            if i in frames.keys():
                frames[i] += dist-threshold
            else :
                frames[i] = dist-threshold
                num += 1
            if i+1 in frames.keys():
                frames[i+1] += dist-threshold
            else :
                frames[i+1] = dist-threshold
                num += 1
        if num >= 20:
            break
    results = sorted(frames.items(),key=lambda x:x[1], reverse=True)
    k = 1
    for key, _ in results:
        pic = cv2.imread(os.path.join(folder_path, files[key]))
        # print(os.path.join(output_path, '{:d}.png'.format(k)))
        cv2.imwrite(os.path.join(output_path, '{:d}.png'.format(k)), pic)
        k += 1
