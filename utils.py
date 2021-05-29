import os
import math

import torch
import matplotlib.pyplot as plt
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


def calculate_giou(box1, box2):
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


def calculate_ciou_loss(box1, box2):

    box1_area = torch.clamp((box1[:, 2] - box1[:, 0]), min=0) * torch.clamp((box1[:, 3] - box1[:, 1]), min=0)
    box2_area = torch.clamp((box2[:, 2] - box2[:, 0]), min=0) * torch.clamp((box2[:, 3] - box2[:, 1]), min=0)

    inter_width = torch.clamp(torch.min(box1[:, 2], box2[:, 2]) - torch.max(box1[:, 0], box2[:, 0]), min=0)
    inter_height = torch.clamp(torch.min(box1[:, 3], box2[:, 3]) - torch.max(box1[:, 1], box2[:, 1]), min=0)
    inter_area = inter_width * inter_height

    iou = inter_area / (box1_area + box2_area - inter_area)

    rate1 = torch.arctan((box1[:, 2] - box1[:, 0]) / (box1[:, 3] - box1[:, 1]))
    rate2 = torch.arctan((box2[:, 2] - box2[:, 0]) / (box2[:, 3] - box2[:, 1]))

    v = (4 / math.pi) * torch.pow(rate1 - rate2, 2)
    alpha = v / (1 - iou + v)

    box1_center_x = torch.clamp((box1[:, 2] + box1[:, 2]) / 2, min=0)
    box1_center_y = torch.clamp((box1[:, 3] + box1[:, 1]) / 2, min=0)

    box2_center_x = torch.clamp((box1[:, 2] + box1[:, 2]) / 2, min=0)
    box2_center_y = torch.clamp((box1[:, 3] + box1[:, 1]) / 2, min=0)

    center_distance = torch.pow(box1_center_x - box2_center_x, 2) + torch.pow(box1_center_y - box2_center_y, 2)

    outer_width = torch.clamp(torch.max(box1[:, 2], box2[:, 2]) - torch.min(box1[:, 0], box2[:, 0]), min=0)
    outer_height = torch.clamp(torch.max(box1[:, 3], box2[:, 3]) - torch.min(box1[:, 1], box2[:, 1]), min=0)
    outer_diagonal_length = torch.pow(outer_width, 2) + torch.pow(outer_height, 2)

    return 1 - iou + center_distance / outer_diagonal_length + alpha * v


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
