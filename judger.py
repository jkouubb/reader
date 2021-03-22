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

import random


class _ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class _MaxPoolLayer(nn.Module):
    def __init__(self, kernel_size, padding, stride):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.max_pool(x)
        x = self.relu(x)
        return x


class Judger(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = _ConvLayer(in_channels=2, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv_layer2 = _ConvLayer(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.max_pool_layer1 = _MaxPoolLayer(kernel_size=3, padding=1, stride=2)
        self.conv_layer3 = _ConvLayer(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv_layer4 = _ConvLayer(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.max_pool_layer2 = _MaxPoolLayer(kernel_size=3, padding=1, stride=2)
        self.conv_layer5 = _ConvLayer(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv_layer6 = _ConvLayer(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.conv_layer7 = _ConvLayer(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.max_pool_layer3 = _MaxPoolLayer(kernel_size=3, padding=1, stride=2)
        self.conv_layer8 = _ConvLayer(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv_layer9 = _ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv_layer10 = _ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.max_pool_layer4 = _MaxPoolLayer(kernel_size=3, padding=1, stride=2)
        self.conv_layer11 = _ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv_layer12 = _ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.conv_layer13 = _ConvLayer(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.max_pool_layer5 = _MaxPoolLayer(kernel_size=3, padding=1, stride=2)
        self.full_connect1 = nn.Linear(in_features=7 * 7 * 512, out_features=4096)
        self.full_connect2 = nn.Linear(in_features=4096, out_features=4096)
        self.full_connect3 = nn.Linear(in_features=4096, out_features=1000)
        self.full_connect4 = nn.Linear(in_features=1000, out_features=1)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.max_pool_layer1(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.max_pool_layer2(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = self.max_pool_layer3(x)
        x = self.conv_layer8(x)
        x = self.conv_layer9(x)
        x = self.conv_layer10(x)
        x = self.max_pool_layer4(x)
        x = self.conv_layer11(x)
        x = self.conv_layer12(x)
        x = self.conv_layer13(x)
        x = self.max_pool_layer5(x)
        x = x.view(x.shape[0], -1)
        x = self.full_connect1(x)
        x = self.full_connect2(x)
        x = self.full_connect3(x)
        x = self.full_connect4(x)

        return x


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
        image1 = image1.transpose((2, 0, 1))
        image1 = (image1[0] + image1[1] + image1[2]) / (3 * 255.0)

        # random_change1 = random.randint(0, 9)
        # random_angle1 = random.randint(0, 3)
        #
        # if random_change1 < 5:
        #     image1 = skimage.transform.rotate(image1, 90 * random_angle1)

        img1_threshold = skimage.filters.threshold_li(image1)
        img1_mask = numpy.zeros_like(image1)
        img1_mask[image1 < img1_threshold] = 1
        img1_mask[image1 > img1_threshold] = 2
        image1 = skimage.segmentation.watershed(image1, img1_mask)
        image1 = image1.astype(numpy.float32)

        image2 = skimage.io.imread('./judger_train_data/{}.jpg'.format(self.index_list[index][1]))
        image2 = skimage.transform.resize(image2, (224, 224, 3))
        image2 = image2.transpose((2, 0, 1))
        image2 = (image2[0] + image2[1] + image2[2]) / (3 * 255.0)

        # random_change2 = random.randint(0, 9)
        # random_angle2 = random.randint(0, 3)
        #
        # if random_change2 < 5:
        #     image2 = skimage.transform.rotate(image2, 90 * random_angle2)

        img2_threshold = skimage.filters.threshold_li(image2)
        img2_mask = numpy.zeros_like(image2)
        img2_mask[image2 < img2_threshold] = 1
        img2_mask[image2 > img2_threshold] = 2
        image2 = skimage.segmentation.watershed(image2, img2_mask)
        image2 = image2.astype(numpy.float32)

        image = torch.stack((torch.from_numpy(image1), torch.from_numpy(image2)))

        # label = [0, 0]
        # label[self.index_list[index][2]] = 1
        label = self.index_list[index][2]

        if label == 0:
            label = -1

        return image, label

    def __len__(self):
        return len(self.index_list)


class JudgerManager:
    def __init__(self, param_path, train_epoch=300, learn_rate=1e-3, batch_size=8):
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

        # optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learn_rate, weight_decay=5e-5)
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.learn_rate, momentum=0.9, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[60, 130, 210, 300, 400], gamma=0.6)

        self.model.train()
        loss_list = []

        for i in range(self.train_epoch):
            average_loss = 0.0

            for batch_index, data in enumerate(train_data_loader):
                images = data[0]
                annotations = data[1]

                optimizer.zero_grad()

                result = self.model(images.cuda())

                loss = torch.clamp(1 - annotations.cuda().float() * result, min=0).mean()

                average_loss += loss

                loss.backward()

                optimizer.step()

            loss_list.append(average_loss / train_data_set.__len__().__float__())

            # scheduler.step()

            print('average loss in epoch {}: {}'.format(i + 1, average_loss / train_data_set.__len__().__float__()))

        epoch_list = [x + 1 for x in range(self.train_epoch)]

        plt.plot(epoch_list, loss_list, color='red', linewidth=2.0, linestyle='--')

        plt.show()

        torch.save(self.model.state_dict(), self.param_path)
        torch.cuda.empty_cache()

    def test(self, image1, image2):
        img1 = skimage.transform.resize(image1, (224, 224, 3))
        img1 = img1.transpose((2, 0, 1))
        img1 = (img1[0] + img1[1] + img1[2]) / (3 * 255.0)
        img1_threshold = skimage.filters.threshold_li(img1)
        img1_mask = numpy.zeros_like(img1)
        img1_mask[img1 < img1_threshold] = 1
        img1_mask[img1 > img1_threshold] = 2
        img1 = skimage.segmentation.watershed(img1, img1_mask)
        img1 = img1.astype(numpy.float32)
        img1 = torch.from_numpy(img1)

        img2 = skimage.transform.resize(image2, (224, 224, 3))
        img2 = img2.transpose((2, 0, 1))
        img2 = (img2[0] + img2[1] + img2[2]) / (3 * 255.0)
        img2_threshold = skimage.filters.threshold_li(img2)
        img2_mask = numpy.zeros_like(img2)
        img2_mask[img2 < img2_threshold] = 1
        img2_mask[img2 > img2_threshold] = 2
        img2 = skimage.segmentation.watershed(img2, img2_mask)
        img2 = img2.astype(numpy.float32)
        img2 = torch.from_numpy(img2)

        img = torch.stack((img1, img2))
        img = torch.unsqueeze(img, dim=0)

        # plt.subplot(1, 2, 1)
        # plt.imshow(img1)
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(img2)
        #
        # plt.show()

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

                rate = self.test(question_tmp, answer_tmp)[0]

                if select_rate < rate:
                    # answer_list.append(answer_tmp_list[j])
                    # used_list.append(j)
                    # break
                    select_index = j
                    select_rate = rate

            answer_list.append(answer_tmp_list[select_index])
            used_list.append(select_index)

        return question_tmp_list, answer_list


def _judger_collate(batch):
    d = [item[0] for item in batch]
    data = torch.stack(d, 0)
    label = [item[1] for item in batch]
    return data, torch.from_numpy(numpy.array(label))
