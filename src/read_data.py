from torch.utils.data import Dataset, DataLoader  # 导入 Dataset 类用于自定义数据集，DataLoader 用于批量加载数据
import numpy as np  # 导入 numpy，用于科学计算
from PIL import Image  # 导入 PIL 库中的 Image 模块，用于图像处理
import os  # 导入 os 模块，用于文件和路径操作
from torchvision import transforms  # 导入 torchvision.transforms，用于对图像进行预处理
from torch.utils.tensorboard import SummaryWriter  # 导入 SummaryWriter，用于将数据写入 TensorBoard 进行可视化
from torchvision.utils import make_grid  # 导入 make_grid，用于将多张图像合成一个图像网格

writer = SummaryWriter("logs")  # 初始化 SummaryWriter，将日志保存到 "logs" 目录中

class MyData(Dataset):  # 定义自定义数据集类 MyData，继承自 PyTorch 的 Dataset
    def __init__(self, root_dir, image_dir, label_dir, transform):
        self.root_dir = root_dir  # 数据集的根目录
        self.image_dir = image_dir  # 图像文件所在的子目录
        self.label_dir = label_dir  # 标签文件所在的子目录
        self.label_path = os.path.join(self.root_dir, self.label_dir)  # 拼接得到标签文件的完整路径
        self.image_path = os.path.join(self.root_dir, self.image_dir)  # 拼接得到图像文件的完整路径
        self.image_list = os.listdir(self.image_path)  # 获取图像文件目录中的所有文件名
        self.label_list = os.listdir(self.label_path)  # 获取标签文件目录中的所有文件名
        self.transform = transform  # 保存传入的图像预处理方法
        self.image_list.sort()  # 对图像文件列表进行排序，确保图像与标签对应
        self.label_list.sort()  # 对标签文件列表进行排序，确保标签与图像对应

    def __getitem__(self, idx):  # 定义获取数据样本的方法
        img_name = self.image_list[idx]  # 获取索引为 idx 的图像文件名
        label_name = self.label_list[idx]  # 获取索引为 idx 的标签文件名
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)  # 获取图像文件的完整路径
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)  # 获取标签文件的完整路径
        img = Image.open(img_item_path)  # 打开图像文件

        with open(label_item_path, 'r') as f:  # 打开标签文件
            label = f.readline()  # 读取标签文件中的内容

        img = self.transform(img)  # 对图像进行预处理（如调整大小，转换为张量）
        sample = {'img': img, 'label': label}  # 将图像和标签打包成一个字典
        return sample  # 返回字典

    def __len__(self):  # 定义返回数据集大小的方法
        assert len(self.image_list) == len(self.label_list)  # 确保图像和标签数量一致
        return len(self.image_list)  # 返回数据集中样本的数量

if __name__ == '__main__':  # 当脚本作为主程序运行时执行以下代码
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])  # 定义图像预处理操作，调整大小并转换为张量
    root_dir = "dataset/train"  # 数据集的根目录
    image_ants = "ants_image"  # 蚂蚁图像所在子目录
    label_ants = "ants_label"  # 蚂蚁标签所在子目录
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)  # 创建蚂蚁数据集实例
    image_bees = "bees_image"  # 蜜蜂图像所在子目录
    label_bees = "bees_label"  # 蜜蜂标签所在子目录
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)  # 创建蜜蜂数据集实例
    train_dataset = ants_dataset + bees_dataset  # 合并蚂蚁和蜜蜂数据集

    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)  # 使用 DataLoader 以批量方式加载数据，batch_size=1，使用2个子进程

    writer.add_image('error', train_dataset[119]['img'])  # 将索引为 119 的图像写入 TensorBoard，标签为 'error'
    writer.close()  # 关闭 SummaryWriter，释放资源

    # for i, j in enumerate(dataloader):  # 遍历 DataLoader 中的数据
    #     # imgs, labels = j
    #     print(type(j))  # 打印每个数据批次的类型
    #     print(i, j['img'].shape)  # 打印当前批次索引和图像的形状
    #     # writer.add_image("train_data_b2", make_grid(j['img']), i)  # 将批次图像合并为网格并写入 TensorBoard
    #
    # writer.close()  # 关闭 SummaryWriter



代码使用：
1. 代码中的数据集部分
数据集部分定义在 MyData 类中，它负责加载指定目录中的图像和标签，并通过索引获取数据。具体操作包括：

使用 __init__ 方法初始化数据集，包括指定图像和标签文件的目录，并进行排序以保证图像和标签一一对应。
使用 __getitem__ 方法根据索引 (idx) 返回相应的图像和标签。
使用 __len__ 方法返回数据集中图像的数量。

2. 使用数据集的实例
假设 ants_dataset 和 bees_dataset 已经使用指定的目录和文件列表进行了初始化，
下面是 dataset 中各个方法的使用以及在你的情况下的结果：

代码示例：使用 dataset 获取数据
已知：
ants 数据集的第 1 个索引对应的文件名是 5650366_e22b7e1065.jpg。
bees 数据集的第 1 个索引对应的文件名是 17209602_fe5a5a746f.jpg。

假设在代码中已经创建了这两个数据集并合并为 train_dataset：
# 获取 ants_dataset 中索引为 1 的数据
sample_ants = ants_dataset[1]
print("Ants Image Path:", os.path.join(root_dir, image_ants, ants_dataset.image_list[1]))
print("Ants Label:", sample_ants['label'])

# 获取 bees_dataset 中索引为 1 的数据
sample_bees = bees_dataset[1]
print("Bees Image Path:", os.path.join(root_dir, image_bees, bees_dataset.image_list[1]))
print("Bees Label:", sample_bees['label'])

解释：代码的工作过程
对于 ants_dataset[1]:
image_list[1] 是排序后的图像文件列表的第 1 个索引，在你的情况下为 "5650366_e22b7e1065.jpg"。
__getitem__ 方法将打开这个图像文件，并应用 transform（如调整大小、转换为张量）。
它还将打开对应的标签文件，读取并返回图像和标签的字典：{'img': transformed_image, 'label': label}。
对于 bees_dataset[1]:
image_list[1] 是 "17209602_fe5a5a746f.jpg"，通过类似的方式读取图像和标签。
预期的输出结果
如果 ants_dataset[1] 对应的文件为 "5650366_e22b7e1065.jpg"，且标签文件的内容为 "ant"，则输出可能为：
Ants Image Path: dataset/train/ants_image/5650366_e22b7e1065.jpg
Ants Label: ant

如果 bees_dataset[1] 对应的文件为 "17209602_fe5a5a746f.jpg"，且标签文件的内容为 "bee"，则输出可能为：
Bees Image Path: dataset/train/bees_image/17209602_fe5a5a746f.jpg
Bees Label: bee







