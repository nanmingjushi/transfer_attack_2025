import os
import sys
import json
import torch
import timm
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from TransferAttack.transferattack import load_attack_class
from TransferAttack.transferattack.utils import AdvDataset, save_images
from TransferAttack.transferattack.utils import wrap_model
import torchvision.models as models
import torch.nn.functional as F

# 1.准备数据集
input_dir = './dataset/imagenet_val_1000'
output_dir = './output/bfa_result'
targeted = False  # 是否进行有目标攻击
target_class = None  # 如果是有目标攻击，指定目标类别
eval_mode = False  # 是否评估模式
# 创建对抗样本数据集对象
dataset = AdvDataset(input_dir=input_dir, output_dir=output_dir, targeted=targeted, target_class=target_class,
                     eval=eval_mode)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # 每次只针对一张图单独生成扰动

# 2.加载攻击类
attack_name = 'bfa'  # 指定攻击方法
AttackClass = load_attack_class(attack_name)  # 加载对应的攻击类

# 3.代理模型
surrogate_model_name = 'resnet18'  # 代理模型
epsilon = 16 / 255  # 扰动幅度
random_start = False  # 是否对扰动进行随机初始化
norm = 'linfty'  # 扰动的范数 支持 'l2', 'linfty'
loss = 'crossentropy'  # 攻击的损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化代理模型的攻击
attack = AttackClass(surrogate_model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., eta=28, num_ens=30,
                 targeted=False, random_start=False, layer_name='layer2.1', norm='linfty', loss='crossentropy', device=None, attack='BFA')

# 4.被攻击模型
target_model_name = 'resnet50'
target_model = models.__dict__[target_model_name](weights="DEFAULT")  # 预训练模型权重在 C:\Users\26354\.cache\torch\hub\checkpoints路径下
target_model = wrap_model(target_model.eval().to(device))

# 5.获取ImageNet标签映射
label_path = './dataset/imagenet_val_1000/imagenet_class_index.json'
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        imagenet_labels = json.load(f)
    imagenet_labels = {int(k): v[1] for k, v in imagenet_labels.items()}
else:
    print(
        f"未找到 {label_path} 文件，请从 https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json 下载该文件。")
    sys.exit(1)

# 6.进行攻击，生成对抗样本
generated_count = 0
success_count=0 # 攻击成功的数量

for images, labels, filenames in data_loader:
    if generated_count >= 50:  # 生成多少个对抗样本
        break
    images = images.to(device)
    labels = labels.to(device)

    # 攻击，生成扰动
    delta = attack(images, labels)

    # 生成对抗样本
    adversarial_images = torch.clamp(images + delta, 0, 1)

    # 获取对抗样本的预测标签和置信度分数
    with torch.no_grad():
        logits = target_model(adversarial_images)
        # 对logits进行softmax操作得到概率分布
        probabilities = F.softmax(logits, dim=1)
        # 获取最大概率值作为置信度分数
        confidences, adversarial_labels = torch.max(probabilities, dim=1)

    # 保存对抗样本
    save_images(output_dir, adversarial_images, filenames, labels, adversarial_labels, targeted, imagenet_labels,
                confidences)

    # 统计攻击成功率
    if targeted:
        # 有目标攻击: 预测 == target_class 才算成功
        success_count += (adversarial_labels == target_class).sum().item()
    else:
        # 无目标攻击: 预测 != 原始标签才算成功
        success_count += (adversarial_labels != labels).sum().item()

    generated_count += images.size(0)


print(f"✅ 已在{surrogate_model_name}上生成 {generated_count} 张对抗样本，并在 {target_model_name} 上测试迁移效果。")

# 欺骗率
attack_success_rate = success_count / generated_count * 100
print(f"⚡ 攻击成功率(欺骗率): {attack_success_rate:.2f}%")
