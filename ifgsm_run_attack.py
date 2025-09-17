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
output_dir = './output/ifgsm_result'
targeted = False  # 是否进行有目标攻击
target_class = None  # 如果是有目标攻击，指定目标类别
eval_mode = False  # 是否评估模式
# 创建对抗样本数据集对象
dataset = AdvDataset(input_dir=input_dir, output_dir=output_dir, targeted=targeted, target_class=target_class, eval=eval_mode)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


# 2.加载攻击类
attack_name = 'ifgsm'  # 指定攻击方法
AttackClass = load_attack_class(attack_name)  # 加载对应的攻击类


# 3.初始化攻击类
model_name = 'resnet50'  # 被攻击模型
epsilon = 16/255  # 扰动幅度
random_start = False  # 是否对扰动进行随机初始化
norm = 'linfty'  # 扰动的范数 支持 'l2', 'linfty'
loss = 'crossentropy'  # 攻击的损失函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化攻击类
attack = AttackClass(model_name, epsilon ,alpha=1.6/255,epoch=10, targeted=targeted, random_start=random_start, norm=norm, loss=loss, device=device,  attack='I-FGSM')

# 加载被攻击的模型
model = models.__dict__[model_name](weights="DEFAULT")  # 预训练模型权重在 C:\Users\26354\.cache\torch\hub\checkpoints路径下
model = wrap_model(model.eval().to(device))
# 获取ImageNet标签映射
label_path = './dataset/imagenet_val_1000/imagenet_class_index.json'
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        imagenet_labels = json.load(f)
    imagenet_labels = {int(k): v[1] for k, v in imagenet_labels.items()}
else:
    print(f"未找到 {label_path} 文件，请从 https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json 下载该文件。")
    sys.exit(1)


# 4.进行攻击
generated_count = 0
for images, labels, filenames in data_loader:
    if generated_count >= 10: # 只生成10个对抗样本
        break
    images = images.to(device)
    labels = labels.to(device)

    # 攻击，生成扰动
    delta = attack(images, labels)

    # 生成对抗样本
    adversarial_images = images + delta

    # 获取对抗样本的预测标签和置信度分数
    with torch.no_grad():
        logits = model(adversarial_images)
        # 对logits进行softmax操作得到概率分布
        probabilities=F.softmax(logits, dim=1)
        # 获取最大概率值作为置信度分数
        confidences, adversarial_labels = torch.max(probabilities, dim=1)


    # # 保存对抗样本
    # save_images(output_dir, adversarial_images, filenames)
    # 保存对抗样本
    save_images(output_dir, adversarial_images, filenames, labels, adversarial_labels, targeted,imagenet_labels,confidences)

    generated_count += 1




