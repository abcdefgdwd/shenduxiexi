
import numpy as np
import os, sys, glob, shutil, json
from torch.optim.lr_scheduler import StepLR  # PyTorch中的学习率调度器，用于动态地调整模型的学习率。
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
from PIL import Image
from tqdm import tqdm, tqdm_notebook  # 用于创建进度条以监视代码中循环的进展。
import torch
torch.manual_seed(0)  # 通过设置随机种子，可以实现重复性，便于调试和验证模型的稳定性。
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms  # 用于图像预处理和数据增强的工具。例如，可以使用这些转换来对图像进行裁剪、缩放、标准化等操作。
import torch.nn as nn
from torch.utils.data.dataset import Dataset


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    train_path = glob.glob('./content/NDataset/mchar_train/*.png')
    train_path.sort()
    train_json = json.load(open('./content/NDataset/mchar_train.json'))
    train_label = [train_json[x]['label'] for x in train_json]

    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                    transforms.Compose([
                        transforms.Resize((64, 128)),
                        transforms.RandomCrop((60, 120)),  # 随即裁剪，有助于模型学习不同部分的特征，同时也可以增加数据的多样性。
                        transforms.ColorJitter(0.3, 0.3, 0.2),  # 颜色抖动操作。这个操作会随机调整图像的亮度、对比度和饱和度，以增加数据的变化性。
                        transforms.RandomRotation(10),  # 随机旋转操作。它会随机旋转图像不超过10度，用于增加数据的多样性和鲁棒性。
                        transforms.ToTensor(),  # 将图像数据转换为张量（Tensor）格式。神经网络模型通常需要输入张量数据。
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化，以便更好地适应模型的训练。
                    ])),
        batch_size=40,
        shuffle=True,
        num_workers=10,
    )

    val_path = glob.glob('./content/NDataset/mchar_val/*.png')
    val_path.sort()
    val_json = json.load(open('./content/NDataset/mchar_val.json'))
    val_label = [val_json[x]['label'] for x in val_json]

    val_loader = torch.utils.data.DataLoader(
        SVHNDataset(val_path, val_label,
                    transforms.Compose([
                        transforms.Resize((60, 120)),
                        #transforms.ColorJitter(0.3, 0.3, 0.2),
                        #transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=False,
        num_workers=10,
    )


class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet50(pretrained=True)  # 会下载并加载在 ImageNet 数据集上预训练过的 ResNet-50模型权重
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)  # 将平均池化替换为自适应平均池化层，为了在不同输入尺寸的图像上使用这个模型
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])  # 去除最后一层分类器，只有卷积层和池化层的ResNet-50
        self.cnn = model_conv
        # 五个全连接层，每一个用于预测一个数字的类别
        self.fc1 = nn.Linear(2048, 11)
        self.fc2 = nn.Linear(2048, 11)
        self.fc3 = nn.Linear(2048, 11)
        self.fc4 = nn.Linear(2048, 11)
        self.fc5 = nn.Linear(2048, 11)

    def forward(self, img):
        feat = self.cnn(img)  # 包含了提取的特征
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)  # 展平
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5

def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):  # 每个批次大小是40
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        target = target.long()
        c0, c1, c2, c3, c4 = model(input)
        loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3]) + \
               criterion(c4, target[:, 4])

        # loss /= 6
        optimizer.zero_grad()  # 将优化器的梯度缓冲区清零，以准备计算新的梯度。
        loss.backward()  # 进行反向传播，计算梯度
        optimizer.step()  # 更新参数，通过梯度下降来最小化损失函数

        train_loss.append(loss.item())  # 将每一个批次的损失加入一个列表中
    return np.mean(train_loss)  # 返回每个批次的损失平均值


def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0, c1, c2, c3, c4 = model(input)
            # 注意：将 target 转换为整数类型的张量
            target = target.long()

            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3]) + \
                   criterion(c4, target[:, 4])
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    # TTA是一种在测试过程中对输入数据进行多次变换或扰动，并对每个变换后的输入进行预测，然后取多次预测的平均值以提高模型性能的技术。
    # TTA 次数
    for _ in range(tta):
        test_pred = []  # 存储每个测试样本的预测结果

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3, c4 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


if __name__ == '__main__':
    # 定义初始学习率
    initial_lr = 0.001
    # 定义初始学习率
    model = SVHN_Model1()
    criterion = nn.CrossEntropyLoss()  # 交叉熵通常用于分类问题，适用于多类别分类任务。
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    # 创建学习率调度器
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    best_loss = 1000.0

    # 是否使用GPU
    use_cuda = True
    if use_cuda:
        model = model.cuda()

    Trainloss = [3.4758074628512063, 2.2395448048909503, 1.8943883774280548, 1.679753142674764, 1.538184202114741, 1.2024751572608947, 1.0937894903421401, 1.0267940393487613, 0.9690206668972969, 0.8927318509618442]
    Valloss = [3.7009251165390014, 3.219884060382843, 2.9071946544647216, 2.7436023941040037, 2.755801487445831, 2.347395691871643, 2.3733014030456543, 2.3956906967163087, 2.429066876173019, 2.4302730967998505]
    Valacc = [0.2736,0.3677,4728,0.4905,0.4936,0.5135,0.5653,0.5697,0.577,0.609]

    for epoch in range(10):
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion)

        # 在每个 epoch 结束时更新学习率
        scheduler.step()

        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
            val_predict_label[:, 44:55].argmax(1),
        ]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
        Trainloss.append(train_loss)
        Valloss.append(val_loss)
        Valacc.append(val_loss)
        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print('Val Acc', val_char_acc)
        # 记录下验证集精度
        if val_loss < best_loss:
            best_loss = val_loss
            # print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), './model.pt')
    print(Trainloss)
    print(Valloss)
    print(Valacc)
    test_path = glob.glob('./content/NDataset/mchar_test_a/*.png')
    test_path.sort()
    # test_json = json.load(open('../input/test_a.json'))
    test_label = [[1]] * len(test_path)
    # print(len(test_path), len(test_label))

    test_loader = torch.utils.data.DataLoader(
        SVHNDataset(test_path, test_label,
                    transforms.Compose([
                        transforms.Resize((70, 140)),
                        # transforms.RandomCrop((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=False,
        num_workers=10,
    )

    # 加载保存的最优模型
    model.load_state_dict(torch.load('model.pt'))

    test_predict_label = predict(test_loader, model, 1)
    # print(test_predict_label.shape)

    test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
    test_predict_label = np.vstack([
        test_predict_label[:, :11].argmax(1),
        test_predict_label[:, 11:22].argmax(1),
        test_predict_label[:, 22:33].argmax(1),
        test_predict_label[:, 33:44].argmax(1),
        test_predict_label[:, 44:55].argmax(1),
    ]).T

    test_label_pred = []
    for x in test_predict_label:
        test_label_pred.append(''.join(map(str, x[x != 10])))

    import pandas as pd

    df_submit = pd.read_csv('content/NDataset/mchar_sample_submit_A.csv')
    df_submit['file_code'] = test_label_pred
    df_submit.to_csv('content/NDataset/submit.csv', index=None)
