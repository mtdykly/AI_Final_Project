# Project-5 : 多模态情感分析

本实验旨在实现一个多模态情感分析模型，处理文本和图像的输入数据，预测对应的情感标签（positive、neutral、negative）。

## 项目设置

此实验基于 Python 3.11 。要运行代码，您需要以下依赖项：

- matplotlib==3.10.0
- pandas==2.2.3
- Pillow==11.1.0
- scikit_learn==1.6.1
- timm==1.0.14
- torch==2.0.1+cu117
- torchvision==0.15.2+cu117
- transformers==4.47.1

您只需运行

```bash
pip install -r requirements.txt
```

## 项目结构

以下是本项目的主要结构：

```txt
|-- dataset # 实验数据集
    |-- data # 包括所有的训练文本和图片，每个文件按照唯一的guid命名
    |-- test_without_label.txt # 测试数据包括数据的guid和空的情感标签
    |-- train.txt # 训练数据包括数据的guid和对应的情感标签
|-- output # 输出文件
	|-- best_model.pth # 保存的最佳模型
	|-- loss_curve.png # 输出的训练和验证损失下降图
|-- utils 
    |-- data_model_utils.py # 公共函数(数据加载、模型训练、模型评估)
|-- decision_level_fusion.py # 决策级融合模型
|-- model_attention.py # 采用跨模态注意力机制和Transformer的模型
|-- model_basic.py # 简单拼接特征并分类的基础模型
|-- model_fc.py # 加入多层全连接层的模型
|-- model_image_only.py # 图像单模态模型
|-- model_text_only.py # 文本单模态模型
|-- model_vit.py # 图像特征提取采用ViT模型的跨模态注意力机制和Transformer模型
```

## 项目运行

您可以通过脚本运行项目文件夹下的所有模型，例如

1. 运行决策级融合模型：

```bash
python decision_level_fusion.py
```

2. 运行采用跨模态注意力机制和Transformer的模型：

```bash
python model_attention.py
```

3. 运行简单拼接特征并分类的基础模型：

```bash
python model_basic.py
```

4. 运行加入多层全连接层的模型：

```bash
python model_fc.py
```

5. 运行图像单模态模型：

```bash
python model_image_only.py
```

6. 运行文本单模态模型：

```bash
python model_text_only.py
```

7. 运行图像特征提取采用ViT模型的跨模态注意力机制和Transformer模型：

```bash
python model_vit.py
```

## 输出文件

1. output/loss_curve.png：展示了训练过程中训练损失和验证损失的变化曲线。
2. output/best_model.pth：保存了验证集上表现最佳的模型参数。
3. dataset/test_without_label.txt：最佳模型在测试集上的预测结果。