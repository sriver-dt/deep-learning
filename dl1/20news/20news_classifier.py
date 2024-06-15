import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Tuple, Any
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']


# 模型测试
class Args:
    def __init__(self):
        self.no_cuda = False
        self.batch_size = 256
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.5
        self.gamma = 0.5
        self.seed = 5
        self.save_model = False
        self.log_interval = 10
        self.dry_run = False
        self.input_feature = 25914    # 25631
        self.hidden_layer = 128
        self.tol = 0.001


class NewsDataset(Dataset):
    def __init__(self, datas, targets):
        self.datas = datas
        self.targets = torch.Tensor(targets)

    # 继承Dataset需要重写此方法，返回一个元组
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.datas[index]
        target = self.targets[index]
        return data, target

    def __len__(self) -> int:
        return len(self.datas)


# 搭建神经网络模型
class NewsNet(nn.Module):
    def __init__(self):
        super(NewsNet, self).__init__()
        args_ = Args()
        # self.batch_norm1d1 = nn.BatchNorm1d(args_.input_feature)
        self.batch_norm1d1 = nn.LayerNorm(args_.input_feature)
        self.linear1 = nn.Linear(args_.input_feature, args_.hidden_layer)
        self.dropout = nn.Dropout(0.25)
        self.linear2 = nn.Linear(args_.hidden_layer, 20)
        # self.batch_norm1d2 = nn.BatchNorm1d(128)
        # self.linear3 = nn.Linear(128, 20)
        # self.batch_norm1d3 = nn.BatchNorm1d(64)
        # self.linear4 = nn.Linear(64, 20)

    def forward(self, x):
        x = self.batch_norm1d1(x)
        x = self.linear1(x)
        x = F.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        # x = self.batch_norm1d2(x)
        # x = F.relu(x)
        # x = self.linear3(x)
        # x = self.batch_norm1d3(x)
        # x = F.relu(x)
        # x = self.linear4(x)
        output = F.log_softmax(x, dim=1)
        return output


# 模型训练
def train(args_, model_: nn.Module, device_, train_loader_, optimizer_, epoch_):
    # 设置模型为 train模式
    model_.train()
    for batch_index, (data, target) in enumerate(train_loader_):
        data, target = data.to(device_), target.to(device_)
        # 梯度清零
        optimizer_.zero_grad()
        # 前向传播
        output = model_(data)
        # 计算损失函数
        loss = F.nll_loss(output, target)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer_.step()

        # 每 10 个 batch 打印一次状态
        if batch_index % args_.log_interval == 0:
            print(f'训练batch: {epoch_} [{batch_index * args_.batch_size}/{len(train_loader_.dataset)} '
                  f'({100.0 * batch_index / len(train_loader_):.2f}%)]  损失: {loss.item():.6f}')
            if args_.dry_run:
                break


def test(model_: nn.Module, device_, test_loader_):
    # 设置模型为 eval模式
    model_.eval()
    test_loss_ = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader_:
            data, target = data.to(device_), target.to(device_)
            output = model_(data)
            # 将一个 batch 的 loss 加起来
            test_loss_ += F.nll_loss(output, target, reduction='sum')
            # 获取最大对数概率的的索引
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    # 计算平均损失
    test_loss_ /= len(test_loader_.dataset)
    accuracy_ = 100.0 * correct / len(test_loader_.dataset)
    # 打印平均损失和准确率
    print(f'测试集: 平均损失: {test_loss_:.6f}, 准确率: {accuracy_:.2f}%')
    return test_loss_.cpu(), accuracy_


def text_vector(train_data_, test_data_):
    # args_ = Args()
    # tf_idf = TfidfVectorizer(max_df=0.5, min_df=5, stop_words='english', max_features=args_.int_feature)
    tf_idf = TfidfVectorizer(max_df=0.5, min_df=5)
    tf_idf.fit(train_data_)
    train_data_ = tf_idf.transform(train_data_)
    test_data_ = tf_idf.transform(test_data_)
    return train_data_, test_data_


def data_preprocessing():
    # remove('headers', 'footers', 'quotes')后的数据
    # train_datas_remove = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    # test_datas_remove = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    # # 原始数据
    train_datas = fetch_20newsgroups(subset='train')
    test_datas = fetch_20newsgroups(subset='test')

    # 文本向量化
    train_data, test_data = text_vector(train_datas.data, test_datas.data)

    train_data, test_data = torch.Tensor(train_data.toarray()), torch.Tensor(test_data.toarray())
    train_target, test_target = torch.LongTensor(train_datas.target), torch.LongTensor(test_datas.target)

    print(train_data.shape)
    # print(train_target[8:20])

    train_data = NewsDataset(train_data, train_target)
    test_data = NewsDataset(test_data, test_target)

    args_ = Args()
    use_cuda = not args_.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args_.seed)
    # device 设置
    if use_cuda:
        device_ = torch.device("cuda:0")
    else:
        device_ = torch.device("cpu")

    train_kwargs = {'batch_size': args_.batch_size}
    test_kwargs = {'batch_size': args_.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader_ = DataLoader(train_data, **train_kwargs)
    test_loader_ = DataLoader(test_data, **test_kwargs)
    return train_loader_, test_loader_, device_


def main():
    # 参数初始化
    args = Args()

    # 数据准备
    train_loader, test_loader, device = data_preprocessing()

    # 创建模型
    model = NewsNet().to(device)

    # 优化器及学习率设置
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    # 调度器, 优化学习率
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    test_loss_accuracy, train_accuracys = [], []
    test_last_accuracy = 0
    # 开始训练
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)

        test_loss, test_accuracy = test(model, device, test_loader)
        test_loss_accuracy.append((test_loss, test_accuracy))

        _, train_accuracy = test(model, device, train_loader)
        train_accuracys.append(train_accuracy)

        scheduler.step()
        if test_accuracy - test_last_accuracy <= args.tol:
            break
        test_last_accuracy = test_accuracy

    # 画损失函数图
    plt.plot(range(len(test_loss_accuracy)), [i[0] for i in test_loss_accuracy], label="损失")
    plt.legend()
    plt.title("测试集损失")
    plt.show()
    plt.plot(range(len(test_loss_accuracy)), [i[1] for i in test_loss_accuracy], label="准确率")
    plt.legend()
    plt.title("测试集准确率")
    plt.show()
    plt.plot(train_accuracys, [i[1] for i in test_loss_accuracy])
    plt.xlabel("训练集准确率")
    plt.ylabel("测试集准确率")
    plt.title("训练集和测试集准确率比较")
    plt.show()


if __name__ == "__main__":
    main()
