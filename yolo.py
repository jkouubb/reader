import os
import xml.dom.minidom

import torch.nn as nn
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

import utils
import skimage.io
import skimage.transform


class _DBL(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.norm = nn.BatchNorm2d(output_channels, eps=1e-3)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky(x)
        return x


class _ResUnit(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.dbl1 = _DBL(input_channels=input_channels, output_channels=int(input_channels / 2), kernel_size=1,
                         padding=0, stride=1)
        self.dbl2 = _DBL(input_channels=int(input_channels / 2), output_channels=input_channels, kernel_size=3,
                         padding=1, stride=1)

    def forward(self, x):
        tmp = self.dbl1(x)
        tmp = self.dbl2(tmp)
        x = x + tmp
        return x


class _ResBlock(nn.Module):
    def __init__(self, input_channels, block_number):
        super().__init__()
        self.down_sample = _DBL(input_channels=input_channels, output_channels=input_channels * 2, kernel_size=3,
                                padding=1, stride=2)
        self.blocks = nn.ModuleList([_ResUnit(input_channels * 2) for i in range(block_number)])

    def forward(self, x):
        x = self.down_sample(x)
        for block in self.blocks:
            x = block(x)
        return x


class _DBLBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.dbl1 = _DBL(input_channels=input_channels, output_channels=output_channels, kernel_size=1,
                         padding=0, stride=1)
        self.dbl2 = _DBL(input_channels=output_channels, output_channels=output_channels * 2, kernel_size=3,
                         padding=1, stride=1)
        self.dbl3 = _DBL(input_channels=output_channels * 2, output_channels=output_channels, kernel_size=1,
                         padding=0, stride=1)
        self.dbl4 = _DBL(input_channels=output_channels, output_channels=output_channels * 2, kernel_size=3,
                         padding=1, stride=1)
        self.dbl5 = _DBL(input_channels=output_channels * 2, output_channels=output_channels, kernel_size=1,
                         padding=0, stride=1)

    def forward(self, x):
        x = self.dbl1(x)
        x = self.dbl2(x)
        x = self.dbl3(x)
        x = self.dbl4(x)
        x = self.dbl5(x)
        return x


class _ConcatLayer(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.dbl = _DBL(input_channels=input_channels, output_channels=int(input_channels / 2), kernel_size=1,
                        padding=0, stride=1)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x2 = self.dbl(x2)
        x2 = self.up_sample(x2)
        return torch.cat((x1, x2), dim=1)


class _OutputLayer(nn.Module):
    def __init__(self, input_channels, class_number):
        super().__init__()
        self.dbl = _DBL(input_channels=input_channels, output_channels=input_channels * 2, kernel_size=3, padding=1,
                        stride=1)
        self.conv = nn.Conv2d(in_channels=input_channels * 2, out_channels=3 * (5 + class_number), kernel_size=1,
                              padding=0, stride=1)

    def forward(self, x):
        x = self.dbl(x)
        x = self.conv(x)
        return x


class YOLO(nn.Module):
    def __init__(self, class_number, small_anchors, middle_anchors, large_anchors):
        super().__init__()

        self.class_number = class_number
        self.small_anchors = small_anchors
        self.middle_anchors = middle_anchors
        self.large_anchors = large_anchors

        self.dbl = _DBL(input_channels=3, output_channels=32, kernel_size=3, padding=1, stride=1)
        self.res_blocks1 = _ResBlock(input_channels=32, block_number=1)
        self.res_blocks2 = _ResBlock(input_channels=64, block_number=2)
        self.res_blocks3 = _ResBlock(input_channels=128, block_number=8)
        self.res_blocks4 = _ResBlock(input_channels=256, block_number=8)
        self.res_blocks5 = _ResBlock(input_channels=512, block_number=4)
        self.dbl_blocks1 = _DBLBlock(input_channels=1024, output_channels=512)
        self.dbl_blocks2 = _DBLBlock(input_channels=768, output_channels=256)
        self.dbl_blocks3 = _DBLBlock(input_channels=384, output_channels=128)
        self.concat_layer1 = _ConcatLayer(input_channels=512)
        self.concat_layer2 = _ConcatLayer(input_channels=256)
        self.output_layer1 = _OutputLayer(input_channels=512, class_number=self.class_number)
        self.output_layer2 = _OutputLayer(input_channels=256, class_number=self.class_number)
        self.output_layer3 = _OutputLayer(input_channels=128, class_number=class_number)

    def forward(self, x):
        x = self.dbl(x)
        x = self.res_blocks1(x)
        x = self.res_blocks2(x)
        x = self.res_blocks3(x)
        output3 = x
        x = self.res_blocks4(x)
        output2 = x
        x = self.res_blocks5(x)
        output1 = x

        output1 = self.dbl_blocks1(output1)
        tmp1 = output1
        output1 = self.output_layer1(output1)

        tmp2 = self.concat_layer1(output2, tmp1)
        tmp2 = self.dbl_blocks2(tmp2)
        output2 = self.output_layer2(tmp2)

        tmp3 = self.concat_layer2(output3, tmp2)
        tmp3 = self.dbl_blocks3(tmp3)
        output3 = self.output_layer3(tmp3)

        output1 = self._decode(output1, self.large_anchors)
        output2 = self._decode(output2, self.middle_anchors)
        output3 = self._decode(output3, self.small_anchors)

        output = torch.cat((output1, output2, output3), dim=1)

        return output  # (batch_size, 10647, [x, y, w, h, p, label1, label2])

    def _decode(self, output, anchors):
        batch_size = output.shape[0]
        grid_size = output.shape[2]
        stride = 416 / grid_size
        anchor_number = len(anchors)
        box_attributes = 5 + self.class_number

        prediction = output.view(batch_size, anchor_number, box_attributes, grid_size, grid_size).permute(0, 1, 3,
                                                                                                          4,
                                                                                                          2).contiguous()

        scaled_anchors = torch.from_numpy(
            np.array([(a_w / stride, a_h / stride) for a_w, a_h in anchors])).float().cuda()

        # scaled_anchors shape（3， 2），3个anchors，每个anchor有w,h两个量。下面步骤是把这两个量划分开
        anchor_w = scaled_anchors[:, 0:1].view((1, anchor_number, 1, 1))  # （param， 3， param， param）
        anchor_h = scaled_anchors[:, 1:2].view((1, anchor_number, 1, 1))  # （param， 3， param， param）

        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4]).cuda()  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:]).cuda()

        pred_boxes = torch.zeros(prediction[..., :4].shape).cuda()
        # 针对每个网格的偏移量，每个网格的单位长度为1，而预测的中心点（x，y）是归一化的（0，1之间），所以可以直接相加
        grid_x = torch.arange(grid_size).repeat(grid_size, 1)
        grid_y = grid_x.t()
        grid_x = grid_x.view([1, 1, grid_size, grid_size]).float().cuda()
        grid_y = grid_y.view([1, 1, grid_size, grid_size]).float().cuda()
        pred_boxes[..., 0] = torch.clamp(x.data + grid_x, min=0, max=416)  # （param, param, gride, gride）
        pred_boxes[..., 1] = torch.clamp(y.data + grid_y, min=0, max=416)
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w  # # （param， 3， param， param）
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # (batch_size, num_anchors*grid_size*grid_size, 85)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride, pred_conf.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.class_number),), -1, )

        return output


class YOLOManager:
    def __init__(self, param_path, class_number, small_anchors, middle_anchors, large_anchors, train_epoch=300, learn_rate=1e-3, batch_size=4, ignore_threshold=0.3,
                 p_threshold=0.85, nms_threshold=0.01):
        self.model = YOLO(class_number, small_anchors, middle_anchors, large_anchors)
        self.param_path = param_path

        if os.path.exists(self.param_path):
            self.model.load_state_dict(torch.load(self.param_path))

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.train_epoch = train_epoch
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.class_number = class_number
        self.ignore_threshold = ignore_threshold
        self.p_threshold = p_threshold
        self.nms_threshold = nms_threshold

    def train(self, train_file_path):

        print('initializing train dataset......')
        train_data_set = YOLODataset(file_path=train_file_path)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_data_set, batch_size=self.batch_size,
                                                        collate_fn=_yolo_collate)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learn_rate, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[20, 90, 150, 220], gamma=0.1)

        print('training......')
        self.model.train()
        loss_list = []
        for i in range(self.train_epoch):
            print('epoch {} in {}'.format(i + 1, self.train_epoch))
            average_loss = 0.0
            for batch_index, data in enumerate(train_data_loader):
                images = data[0]
                annotations = data[1]

                optimizer.zero_grad()

                output = self.model(images.cuda())

                loss = torch.tensor(0).cuda()
                for j in range(self.batch_size):
                    loss = loss + self._calculate_loss(output[j], annotations[j])

                average_loss += loss.data.item()

                loss.backward()

                optimizer.step()

            loss_list.append(average_loss / train_data_set.__len__().__float__())

            scheduler.step()
            print('average loss in epoch {}: {}'.format(i + 1, average_loss / train_data_set.__len__().__float__()))

        epoch_list = [x + 1 for x in range(self.train_epoch)]

        plt.plot(epoch_list, loss_list, color='red', linewidth=2.0, linestyle='--')

        plt.show()

        print('end train......')

        torch.save(self.model.state_dict(), self.param_path)
        torch.cuda.empty_cache()

    def test(self, test_file_path, judger_manager):
        print('initializing test dataset......')
        test_data_set = YOLODataset(file_path=test_file_path)
        test_data_loader = torch.utils.data.DataLoader(dataset=test_data_set, batch_size=self.batch_size,
                                                       collate_fn=_yolo_collate)

        print('testing......')
        self.model.eval()
        for batch_index, data in enumerate(test_data_loader):
            images = data[0]
            annotations = data[1]

            output = self.model(images.cuda())

            predictions = self._make_prediction(output)
            if predictions is None:
                continue
            for j in range(self.batch_size):
                results = []
                for prediction in predictions:
                    if prediction[0] == j:
                        tmp = [prediction[1], prediction[2], prediction[3], prediction[4]]
                        if prediction[5] == 0:
                            tmp.append('answer')
                        elif prediction[5] == 1:
                            tmp.append('question')
                        tmp.append(prediction[6])
                        results.append(tmp)

                # utils.create_judger_train_data(annotations[j].path, results)

                questions, answers = judger_manager.judge_result('./data/images/{}.jpg'.format(annotations[j].path), results)

                utils.mark_results(annotations[j].path, questions, answers)

        print('end test')

    def _calculate_loss(self, predictions, annotations):
        bce = nn.BCELoss()

        # 将(x, y, w, h)转为(x_min, y_min, x_max, y_max)
        tmp = predictions.new(predictions.shape)
        tmp[:, 0] = (predictions[:, 0] - predictions[:, 2] / 2)
        tmp[:, 1] = (predictions[:, 1] - predictions[:, 3] / 2)
        tmp[:, 2] = (predictions[:, 0] + predictions[:, 2] / 2)
        tmp[:, 3] = (predictions[:, 1] + predictions[:, 3] / 2)
        predictions[:, 0:4] = tmp[:, 0:4]
        del tmp

        # annotations转为Tensor
        annotations_tensor = torch.from_numpy(np.array(annotations.objects, dtype=np.float32)).cuda()

        # prediction的类别: 正例(0~class_number)、负例(-1)和忽略(-2)
        match_matrix = -1 * torch.ones(predictions.shape[0])

        # 遍历annotations, 找到负责预测它的网格输出
        for i in range(len(annotations_tensor)):
            annotation_tmp = annotations_tensor[i].repeat((len(predictions), 1)).cuda()

            # 计算iou, 取iou值最大的预测anchor作为负责预测此annotation的anchor
            iou_matrix = utils.calculate_giou(predictions, annotation_tmp)

            # 先更新忽略例
            ignore_mask_tmp = torch.gt(iou_matrix, self.ignore_threshold)
            match_matrix[ignore_mask_tmp] = -2

        for i in range(len(annotations_tensor)):
            annotation_tmp = annotations_tensor[i].repeat((len(predictions), 1)).cuda()

            # 计算iou, 取iou值最大的预测anchor作为负责预测此annotation的anchor
            iou_matrix = utils.calculate_giou(predictions, annotation_tmp)

            max_iou_value, max_iou_index = torch.max(iou_matrix, dim=0)
            match_matrix[max_iou_index] = i

        # 根据对应关系, 可以得到正例和负例的掩码
        positive_mask = torch.gt(match_matrix, -1)
        negative_mask = torch.eq(match_matrix, -1)

        # 通过掩码将得到anchor
        positive_predictions = predictions[positive_mask]
        negative_predictions = predictions[negative_mask]

        # 构造annotation和label矩阵进行后续的误差计算
        annotations_match = torch.zeros((len(positive_predictions), annotations_tensor.shape[1])).cuda()
        label_matrix = torch.zeros((len(positive_predictions), self.class_number)).cuda()
        for i in range(len(match_matrix[positive_mask])):
            annotations_match[i] = annotations_tensor[match_matrix[positive_mask][i].long()]
            label_matrix[i][annotations_match[i][4].long()] = 1

        box_loss = utils.calculate_ciou_loss(positive_predictions, annotations_match).sum()
        classify_loss = bce(positive_predictions[:, 5:5 + self.class_number], label_matrix).sum()
        positive_object_loss = (-1 * torch.log(torch.clamp(positive_predictions[:, 4], min=1e-15))).sum()

        negative_object_loss = (-1 * torch.log(torch.clamp(1 - negative_predictions[:, 4], min=1e-15))).sum()

        return box_loss + classify_loss + positive_object_loss + negative_object_loss

    def _make_prediction(self, output):
        prediction = None

        # (x, y, w, h)转换为(x_min, y_min, x_max, y_max)
        tmp = output.new(output.shape)
        tmp[:, :, 0] = (output[:, :, 0] - output[:, :, 2] / 2)
        tmp[:, :, 1] = (output[:, :, 1] - output[:, :, 3] / 2)
        tmp[:, :, 2] = (output[:, :, 0] + output[:, :, 2] / 2)
        tmp[:, :, 3] = (output[:, :, 1] + output[:, :, 3] / 2)
        output[:, :, 0:4] = torch.clamp(tmp[:, :, 0:4], min=0)

        # 遍历每张图片的输出
        for b in range(self.batch_size):
            output_tmp = output[b]

            # 筛选出p值超过阈值的输出
            p_mask = torch.gt(output_tmp[:, 4], self.p_threshold)
            output_tmp = output_tmp[p_mask]

            # 将筛选后的输出按照p降序排列
            output_sort_index = torch.sort(output_tmp[:, 4], descending=True)[1]

            # 按照类别遍历筛选后的输出
            for i in range(self.class_number):
                # nms后的输出
                tmp_list = None
                for j in range(len(output_sort_index)):
                    single_output = output_tmp[output_sort_index[j]]

                    # 找到该输出预测的类比
                    predict_label = torch.max(single_output[5:5 + self.class_number], dim=0)[1]

                    # 如果不是当前类别, 跳过后续步骤
                    if predict_label != i:
                        continue

                    if tmp_list is None:
                        tmp_list = torch.empty((1, len(single_output))).cuda()
                        tmp_list[0, :] = single_output
                        continue

                    single_output_copy = single_output.repeat((len(tmp_list), 1))
                    iou_vector = utils.calculate_giou(single_output_copy, tmp_list)

                    if torch.max(iou_vector) < self.nms_threshold:
                        tmp = torch.empty((1, len(single_output))).cuda()
                        tmp[0, :] = single_output
                        tmp_list = torch.cat((tmp_list, tmp), dim=0)

                # 添加到最终结果中
                if tmp_list is None:
                    continue
                for j in range(len(tmp_list)):
                    if prediction is None:
                        prediction = torch.zeros((1, 7))
                        prediction[0, 0] = b
                        prediction[0, 1:5] = tmp_list[j, 0:4]
                        prediction[0, 5] = i
                        prediction[0, 6] = tmp_list[j, 5 + i]
                    else:
                        tmp = torch.zeros((1, 7))
                        tmp[0, 0] = b
                        tmp[0, 1:5] = tmp_list[j, 0:4]
                        tmp[0, 5] = i
                        tmp[0, 6] = tmp_list[j, 5 + i]
                        prediction = torch.cat((prediction, tmp), dim=0)

        return prediction  # (batch_index, x_min, y_min, x_max, y_max, label, possibility)


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, file_path, ):
        self.index_list = []
        data_file = open(file_path, 'r')
        for line in data_file.readlines():
            index = line.split('\n')[0]
            self.index_list.append(int(index))

    def __getitem__(self, index):
        image = skimage.io.imread('./data/images/img_{}.jpg'.format(self.index_list[index]))
        image = skimage.transform.resize(image, (416, 416, 3))
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)

        label = self._parse_xml(index)
        return torch.from_numpy(image), label

    def __len__(self):
        return len(self.index_list)

    def _parse_xml(self, index):
        label = YOLOAnnotation('img_{}'.format(self.index_list[index]))

        if os.path.exists('./data/labels/{}.xml'.format('img_{}'.format(self.index_list[index]))):
            dom_tree = xml.dom.minidom.parse('./data/labels/{}.xml'.format('img_{}'.format(self.index_list[index])))
            collection = dom_tree.documentElement
            objects = collection.getElementsByTagName('object')
            for o in objects:
                tmp = [0, 0, 0, 0, 0]
                obj = o.getElementsByTagName('bndbox')[0]
                name = o.getElementsByTagName('name')[0]

                if name.childNodes[0].data == '验证码文字':
                    tmp[4] = 0.0
                elif name.childNodes[0].data == '命令文字':
                    tmp[4] = 1.0

                tmp[0] = float(obj.getElementsByTagName('xmin')[0].childNodes[0].data)
                tmp[1] = float(obj.getElementsByTagName('ymin')[0].childNodes[0].data)
                tmp[2] = float(obj.getElementsByTagName('xmax')[0].childNodes[0].data)
                tmp[3] = float(obj.getElementsByTagName('ymax')[0].childNodes[0].data)
                label.objects.append(tmp)

        return label


class YOLOAnnotation:
    def __init__(self, path):
        self.path = path
        self.objects = []


def _yolo_collate(batch):
    d = [item[0] for item in batch]
    data = torch.stack(d, 0)
    label = [item[1] for item in batch]
    return data, label
