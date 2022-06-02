from model import *
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import cv2
import torch
from data_process import *
import pickle
import torch.nn.functional as F
from tqdm import tqdm
import time
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
import multiprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

root_path = "../ffpp"
output_path = "../ffpp_model"

#训练函数
def train(epochs):
    model.train() #模型设置成训练模式
    for epoch in range(epochs): #训练epochs轮
        loss_sum = 0  #记录每轮loss
        correct = 0
        for batch in train_iter:
            x1, x2, label = batch
            optimizer.zero_grad() #每次迭代前设置grad为0

            output = model(x1, x2)
            #print(output)
            #print(label)

            loss = F.binary_cross_entropy_with_logits(output.squeeze(1), label.float()) #计算loss
            loss.backward() #反向传播
            optimizer.step() #更新模型参数
            loss_sum += loss.item() #累积loss
            predicted = torch.max(output.data.sigmoid(), 1)[0]
            acc = accuracy_score(label.numpy(), predicted.numpy()>0.5)
            print("acc", acc)
        print('epoch: ', epoch, 'loss:', loss_sum / len(train_iter), 'acc:', correct / len(train_iter))


#利用训练好的模型选帧
def evaluate(data, if_fake, fake_type):
    model.eval()
    with torch.no_grad(): #测试时不计算梯度
        if if_fake == 0:
            for folder in os.listdir(os.path.join(root_path, data, "real")):  # process the real dataset
                folder_path = os.path.join(root_path, data, "real", folder)  # a video folder
                os.makedirs(os.path.join(output_path, data, "real", folder), exist_ok=True)
                images = os.listdir(folder_path)  # the list of frames
                if len(images) < 10:
                    continue
                pre = []
                fra = []
                print(folder_path)
                for i in tqdm(range(len(images) - 1)):
                    ori1 = cv2.imread(os.path.join(folder_path, images[i]))
                    ori2 = cv2.imread(os.path.join(folder_path, images[i + 1]))
                    if ori1 is None or ori2 is None:
                        continue
                    PIC1 = cv2.resize(ori1, (160, 160), interpolation=cv2.INTER_CUBIC)
                    PIC2 = cv2.resize(ori2, (160, 160), interpolation=cv2.INTER_CUBIC)
                    pic1 = np.transpose(PIC1, (2, 0, 1))
                    pic1 = torch.tensor(pic1, dtype=torch.float).to("cuda")
                    pic2 = np.transpose(PIC2, (2, 0, 1))
                    pic2 = torch.tensor(pic2, dtype=torch.float).to("cuda")
                    dataset = [(pic1, pic2)]
                    pic1 = DataLoader(dataset, batch_size=1, shuffle=True)
                    for X1, X2 in pic1:
                        predict = model(X1, X2).sigmoid()
                        pre.append(predict)
                        fra.append(i)
                        fra.append(i+1)
                sorted_id = sorted(range(len(pre)), key=lambda k: pre[k], reverse=True)
                k = 1
                already = []
                for id in sorted_id:
                    if fra[id*2] not in already:
                        ori1 = cv2.imread(os.path.join(folder_path, images[fra[id*2]]))
                        cv2.imwrite(os.path.join(output_path, data, "real", folder, '{:d}.png'.format(k)),ori1)
                        k += 1
                        already.append(fra[id*2])
                    if fra[id*2+1] not in already:
                        ori2 = cv2.imread(os.path.join(folder_path, images[fra[id*2+1]]))
                        cv2.imwrite(os.path.join(output_path, data, "real", folder, '{:d}.png'.format(k)),ori2)
                        k += 1
                        already.append(fra[id*2+1])
                    if k > 20:
                        break

        if if_fake == 1:
                for folder in os.listdir(os.path.join(root_path, data, "fake", fake_type)):  # process the fake dataset
                    folder_path = os.path.join(root_path, data, "fake", fake_type, folder)  # a video folder
                    os.makedirs(os.path.join(output_path, data, "fake", fake_type, folder), exist_ok=True)
                    images = os.listdir(folder_path)  # the list of frames
                    if len(images) < 10:
                        continue
                    pre = []
                    fra = []
                    for i in range(len(images) - 1):
                        ori1 = cv2.imread(os.path.join(folder_path, images[i]))
                        ori2 = cv2.imread(os.path.join(folder_path, images[i + 1]))
                        if ori1 is None or ori2 is None:
                            continue
                        PIC1 = cv2.resize(ori1, (160, 160), interpolation=cv2.INTER_CUBIC)
                        PIC2 = cv2.resize(ori2, (160, 160), interpolation=cv2.INTER_CUBIC)
                        pic1 = np.transpose(PIC1, (2, 0, 1))
                        pic1 = torch.tensor(pic1, dtype=torch.float).to("cuda")
                        pic2 = np.transpose(PIC2, (2, 0, 1))
                        pic2 = torch.tensor(pic2, dtype=torch.float).to("cuda")
                        dataset = [(pic1, pic2)]
                        pic1 = DataLoader(dataset, batch_size=1, shuffle=True)
                        for X1, X2 in pic1:
                            predict = model(X1, X2).sigmoid()
                            pre.append(predict)
                            fra.append(i)
                            fra.append(i + 1)
                    sorted_id = sorted(range(len(pre)), key=lambda k: pre[k], reverse=False)
                    k = 1
                    already = []
                    for id in sorted_id:
                        if fra[id * 2] not in already:
                            ori1 = cv2.imread(os.path.join(folder_path, images[fra[id * 2]]))
                            cv2.imwrite(os.path.join(output_path, data, "fake", fake_type, folder, '{:d}.png'.format(k)), ori1)
                            k += 1
                            already.append(fra[id * 2])
                        if fra[id * 2 + 1] not in already:
                            ori2 = cv2.imread(os.path.join(folder_path, images[fra[id * 2 + 1]]))
                            cv2.imwrite(os.path.join(output_path, data, "fake", fake_type, folder, '{:d}.png'.format(k)), ori2)
                            k += 1
                            already.append(fra[id * 2 + 1])
                        if k > 20:
                            break


'''
TORCH_SEED = 21 #随机数种子
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #设置模型在几号GPU上跑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #设置device

# 设置随机数种子，保证结果一致
os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
np.random.seed(TORCH_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dataset = MyDataset("./model_choose_data.npy")

#创建数据集
#train_dataset = MyDataset('./data/acsa_train.json')
#test_dataset = MyDataset('./data/acsa_test.json')
train_iter = DataLoader(train_dataset, batch_size=25, shuffle=True)
#test_iter = DataLoader(test_dataset, batch_size=25, shuffle=False, collate_fn=batch_process)

#定义模型
model = resnet18(1)

#定义loss函数、优化器
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.001)

#开始训练
#train(2)
#f = open("./1.pickle", "wb")
#pickle.dump(model, f)
'''
f = open("./1.pickle", "rb")
model = pickle.load(f).to("cuda")


dataset_type = ['train', 'valid', 'test']
fake = ['Deepfakes', 'Face2Face', 'FaceSwap']



# evaluate('train', 1, 'Deepfakes')
evaluate('train', 1, 'Face2Face')
# evaluate('train', 1, 'FaceSwap')
#
# evaluate('valid', 1, 'Deepfakes')
# evaluate('valid', 1, 'Face2Face')
# evaluate('valid', 1, 'FaceSwap')
# evaluate('train', 0, 0)
# evaluate('valid', 0, 0)

