import os

import torch
import torch.nn.functional
import matplotlib.pyplot as plt
import skimage.transform
import skimage.color
import skimage.filters
import skimage.io

import numpy as np
from sklearn.cluster import KMeans


def k_means_anchor(dataset):
    tmp_list = []
    for index, image in enumerate(dataset):
        label = image[1]
        for obj in label.objects:
            tmp_list.append([obj[2] - obj[0], obj[3] - obj[1]])

    tmp_numpy_list = np.array(tmp_list)
    result = KMeans(n_clusters=9, max_iter=500, tol=1e-6).fit(tmp_numpy_list)

    print(result.cluster_centers_)


def calculate_iou(box1, box2):
    """
    计算两个矩形的iou值
    :param box1: 矩形A
    :param box2: 矩形B
    :return: iou值
    """

    box1_area = torch.clamp((box1[:, 2] - box1[:, 0]), min=0) * torch.clamp((box1[:, 3] - box1[:, 1]), min=0)
    box2_area = torch.clamp((box2[:, 2] - box2[:, 0]), min=0) * torch.clamp((box2[:, 3] - box2[:, 1]), min=0)

    inter_width = torch.clamp(torch.min(box1[:, 2], box2[:, 2]) - torch.max(box1[:, 0], box2[:, 0]), min=0)
    inter_height = torch.clamp(torch.min(box1[:, 3], box2[:, 3]) - torch.max(box1[:, 1], box2[:, 1]), min=0)
    inter_area = inter_width * inter_height

    outer_width = torch.clamp(torch.max(box1[:, 2], box2[:, 2]) - torch.min(box1[:, 0], box2[:, 0]), min=0)
    outer_height = torch.clamp(torch.max(box1[:, 3], box2[:, 3]) - torch.min(box1[:, 1], box2[:, 1]), min=0)
    outer_area = outer_width * outer_height

    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou - ((outer_area - (box1_area + box2_area - inter_area)) / outer_area)


def create_judger_train_data(image_path, results):
    files = os.listdir('./judger_train_data')

    current_length = len(files)

    image = plt.imread('./data/images/{}.jpg'.format(image_path))
    for i in range(len(results)):
        tmp = image[torch.floor(results[i][1]).int():torch.floor(results[i][3]).int(), torch.floor(results[i][0]).int(): torch.floor(results[i][2]).int()]
        skimage.io.imsave('./judger_train_data/{}.jpg'.format(current_length + i + 1), tmp)


def mark_results(image_path, questions, answers):
    image = skimage.io.imread('./data/images/{}.jpg'.format(image_path))

    plt.imshow(image)
    ax = plt.gca()

    for i in range(len(questions)):
        width = questions[i][2] - questions[i][0]
        height = questions[i][3] - questions[i][1]
        ax.add_patch(
            plt.Rectangle((questions[i][0], questions[i][1]),
                          width,
                          height, fill=False,
                          edgecolor='red')
        )
        plt.text(questions[i][0], questions[i][1], '{}'.format(i), color='white', bbox=dict(facecolor='blue'))

    for i in range(len(answers)):
        width = answers[i][2] - answers[i][0]
        height = answers[i][3] - answers[i][1]
        ax.add_patch(
            plt.Rectangle((answers[i][0], answers[i][1]),
                          width,
                          height, fill=False,
                          edgecolor='red')
        )
        plt.text(answers[i][0], answers[i][1], '{}'.format(i), color='white', bbox=dict(facecolor='blue'))

    plt.savefig('./output/{}.jpg'.format(image_path))
    plt.show()
