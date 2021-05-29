import os

import torch.nn as nn
import torch.utils.data
import numpy

import skimage.io
import skimage.transform
import skimage.color
import skimage.filters
import skimage.segmentation
import skimage.feature

import matplotlib.pyplot as plt


class _BasicBlock(nn.Module):
    extend = 1

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(_BasicBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.down_sample = down_sample
        self.relu = nn.LeakyReLU(True)

    def forward(self, x):
        tmp = x
        out = self.layer(x)

        if self.down_sample is not None:
            tmp = self.down_sample(x)

        out = out + tmp
        out = self.relu(out)

        return out


class Judger(nn.Module):
    def __init__(self,):
        super(Judger, self).__init__()

        self.in_channel = 64

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer2 = self._build_block(64, 2)
        self.layer3 = self._build_block(128, 2, 2)
        self.layer4 = self._build_block(256, 2, 2)
        self.layer5 = self._build_block(512, 2, 2)

        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.full_connect = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.average_pool(x)
        x = torch.flatten(x, 1)
        x = self.full_connect(x)
        x = self.sigmoid(x)

        return x

    def _build_block(self, out_channel, block_num, stride=1):
        down_sample = None
        if stride != 1 or self.in_channel != out_channel * _BasicBlock.extend:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * _BasicBlock.extend, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * _BasicBlock.extend)
            )

        layers = [_BasicBlock(self.in_channel, out_channel, stride, down_sample)]

        self.in_channel = out_channel * _BasicBlock.extend
        for i in range(block_num-1):
            layers.append(_BasicBlock(self.in_channel, out_channel))

        return nn.Sequential(*layers)


class JudgerDataSet(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.index_list = []
        data_file = open(file_path, 'r')
        for line in data_file.readlines():
            index = line.split('\n')[0]
            self.index_list.append([int(index.split(',')[0]), int(index.split(',')[1]), int(index.split(',')[2])])

    def __getitem__(self, index):
        image1 = skimage.io.imread('./judger_train_data/{}.jpg'.format(self.index_list[index][0]))
        image1 = skimage.transform.resize(image1, (224, 224, 3))
        image1 = skimage.color.rgb2gray(image1)
        img1_1 = skimage.filters.gaussian(image1, sigma=2)
        img1_2 = skimage.filters.gaussian(image1, sigma=1)
        image1 = (img1_1 - img1_2).astype(numpy.float32)
        image1 = torch.from_numpy(image1)

        image2 = skimage.io.imread('./judger_train_data/{}.jpg'.format(self.index_list[index][1]))
        image2 = skimage.transform.resize(image2, (224, 224, 3))
        image2 = skimage.color.rgb2gray(image2)
        img2_1 = skimage.filters.gaussian(image2, sigma=2)
        img2_2 = skimage.filters.gaussian(image2, sigma=1)
        image2 = (img2_1 - img2_2).astype(numpy.float32)
        image2 = torch.from_numpy(image2)

        image = torch.stack((image1, image2))

        label = [self.index_list[index][2]]

        return image, label

    def __len__(self):
        return len(self.index_list)


class JudgerManager:
    def __init__(self, param_path, train_epoch=300, learn_rate=1e-1, batch_size=8):
        self.model = Judger()

        self.param_path = param_path

        if os.path.exists(self.param_path):
            self.model.load_state_dict(torch.load(self.param_path))

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.train_epoch = train_epoch
        self.learn_rate = learn_rate
        self.batch_size = batch_size

    def train(self, train_file_path):
        train_data_set = JudgerDataSet(file_path=train_file_path)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_data_set, batch_size=self.batch_size,
                                                        collate_fn=_judger_collate)

        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        optimizer = torch.optim.SGD([
          {'params': weight_p, 'weight_decay': 5e-4},
          {'params': bias_p, 'weight_decay': 0}
          ], lr=self.learn_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 100], gamma=0.1, last_epoch=-1)

        self.model.train()
        loss_list = []

        bce = torch.nn.BCELoss()

        for i in range(self.train_epoch):
            average_loss = 0.0

            for batch_index, data in enumerate(train_data_loader):
                images = data[0]
                annotations = data[1]

                optimizer.zero_grad()

                result = self.model(images.cuda())

                loss = bce(result, annotations.float().cuda()).sum()

                average_loss += loss

                loss.backward()

                optimizer.step()

            loss_list.append(average_loss / train_data_set.__len__().__float__())

            scheduler.step()

            print('average loss in epoch {}: {}'.format(i + 1, average_loss / train_data_set.__len__().__float__()))

        epoch_list = [x + 1 for x in range(self.train_epoch)]

        plt.plot(epoch_list, loss_list, color='red', linewidth=2.0, linestyle='--')

        plt.show()

        torch.save(self.model.state_dict(), self.param_path)
        torch.cuda.empty_cache()

    def test(self, image1, image2):

        img1 = skimage.transform.resize(image1, (224, 224, 3))
        img1 = skimage.color.rgb2gray(img1)
        img1_1 = skimage.filters.gaussian(img1, sigma=2)
        img1_2 = skimage.filters.gaussian(img1, sigma=1)
        img1 = (img1_1 - img1_2).astype(numpy.float32)
        img1 = torch.from_numpy(img1)

        img2 = skimage.transform.resize(image2, (224, 224, 3))
        img2 = skimage.color.rgb2gray(img2)
        img2_1 = skimage.filters.gaussian(img2, sigma=2)
        img2_2 = skimage.filters.gaussian(img2, sigma=1)
        img2 = (img2_1 - img2_2).astype(numpy.float32)
        img2 = torch.from_numpy(img2)

        img = torch.stack((img1, img2))
        img = torch.unsqueeze(img, 0)

        self.model.eval()

        result = self.model(img.cuda())

        return result

    def judge_result(self, image_path, results):
        image = skimage.io.imread(image_path)

        question_tmp_list = []
        answer_tmp_list = []

        for result in results:
            if result[4] == 'question':
                question_tmp_list.append(result)
            elif result[4] == 'answer':
                answer_tmp_list.append(result)

        for i in range(len(question_tmp_list)):
            for j in range(i + 1, len(question_tmp_list)):
                if question_tmp_list[i][0] > question_tmp_list[j][0]:
                    tmp = question_tmp_list[i]
                    question_tmp_list[i] = question_tmp_list[j]
                    question_tmp_list[j] = tmp

        answer_list = []
        used_list = []

        for i in range(len(question_tmp_list)):
            question_tmp = image[torch.floor(question_tmp_list[i][1]).int():torch.floor(question_tmp_list[i][3]).int(),
                           torch.floor(question_tmp_list[i][0]).int(): torch.floor(question_tmp_list[i][2]).int()]

            select_index = -1
            select_rate = -10000

            for j in range(len(answer_tmp_list)):
                if used_list.__contains__(j):
                    continue

                answer_tmp = image[torch.floor(answer_tmp_list[j][1]).int():torch.floor(answer_tmp_list[j][3]).int(),
                             torch.floor(answer_tmp_list[j][0]).int(): torch.floor(answer_tmp_list[j][2]).int()]

                rate = self.test(answer_tmp, question_tmp)[0]

                if select_rate < rate:
                    select_index = j
                    select_rate = rate

            if select_index != -1:
                answer_list.append(answer_tmp_list[select_index])
                used_list.append(select_index)

        return question_tmp_list, answer_list


def _judger_collate(batch):
    d1 = [item[0] for item in batch]
    data = torch.stack(d1, 0)

    label = [item[1] for item in batch]
    return data, torch.from_numpy(numpy.array(label))
