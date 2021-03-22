import yolo
import judger


import torch.utils.data
import skimage.io
import skimage.transform


def resize_image(end):
    for i in range(end):
        image = skimage.io.imread('./data/images/img_{}.jpg'.format(i + 1))
        image = skimage.transform.resize(image, (416, 416, 3))
        skimage.io.imsave('./data/images/img_{}.jpg'.format(i + 1), image)


if __name__ == '__main__':

    print('initializing yolo v3......')
    yolo_manager = yolo.YOLOManager(param_path='yolo_param', train_epoch=50, learn_rate=1e-8, class_number=2, small_anchors=[[25, 25], [29, 28], [34, 32]], middle_anchors=[[63, 59], [74, 59], [67, 68]], large_anchors=[[75, 67], [79, 73], [86, 67]])

    judger_manager = judger.JudgerManager(param_path='judger_param', train_epoch=50, learn_rate=1e-5)
    with torch.autograd.set_detect_anomaly(True):
        # yolo_manager.train(train_file_path='yolo_train.txt')
        judger_manager.train(train_file_path='judger_train.txt')

    with torch.no_grad():
        yolo_manager.test(test_file_path='yolo_test.txt', judger_manager=judger_manager)
