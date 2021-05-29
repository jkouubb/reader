import torch
import argparse

import yolo
import judger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    arg = parser.parse_args()

    yolo_manager = yolo.YOLOManager(param_path='yolo_param', train_epoch=100, learn_rate=1e-4, class_number=2,
                                    small_anchors=[[25, 25], [29, 28], [34, 32]],
                                    middle_anchors=[[63, 59], [74, 59], [67, 68]],
                                    large_anchors=[[75, 67], [79, 73], [86, 67]])

    judger_manager = judger.JudgerManager(param_path='judger_param', train_epoch=50, learn_rate=1e-5)

    if arg.train:
        with torch.autograd.set_detect_anomaly(True):
            yolo_manager.train(train_file_path='yolo_train.txt', )
            judger_manager.train(train_file_path='judger_train.txt')

    with torch.no_grad():
        yolo_manager.test(test_file_path='yolo_test.txt', judger_manager=judger_manager)
