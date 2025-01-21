import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import BertTokenizer
from transformers import BertModel
from torchvision import models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签映射
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

# 标签反映射
reverse_label_map = {0: "negative", 1: "neutral", 2: "positive"}

# 数据集加载与预处理
class MultimodalDataset(Dataset):
    def __init__(self, data_dir, data_file, tokenizer, transform=None):
        self.df = pd.read_csv(data_file)
        self.data_dir = data_dir
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        guid = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]

        # 将标签映射为整数
        if label:
            label = label_map.get(label, -1)
            if label == -1:
                raise ValueError(f"Invalid label found: {label}")

        # 读取文本文件
        text_file = os.path.join(self.data_dir, f"{int(guid)}.txt")
        with open(text_file, 'r', encoding='latin1') as file:
            text = file.read().strip()
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

        # 读取图像文件
        img_file = os.path.join(self.data_dir, f"{int(guid)}.jpg")
        img = Image.open(img_file)
        img = self.transform(img)

        # 训练集返回标签，测试集返回guid
        if label == -1:
            return encoding, img, guid
        else:
            return encoding, img, label

# 提取文本特征
def get_text_features(encoding):
    input_ids = encoding['input_ids'].squeeze(1).to(device)
    attention_mask = encoding['attention_mask'].squeeze(1).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        text_features = bert_model(input_ids, attention_mask=attention_mask).pooler_output
    # [batch_size, hidden_dim]
    return text_features

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for text_inputs, img_inputs, labels in train_loader:
        labels = labels.to(device)
        text_inputs = {key: val.squeeze(1).to(device) for key, val in text_inputs.items()}
        img_inputs = img_inputs.to(device)

        optimizer.zero_grad()
        outputs = model(text_inputs, img_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    train_loss /= len(train_loader)
    accuracy = 100 * correct_preds / total_preds
    print(f'Epoch [{epoch + 1}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return train_loss, accuracy

criterion = nn.CrossEntropyLoss()

# 验证函数
def evaluate(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for text_inputs, img_inputs, labels in val_loader:
            labels = labels.to(device)
            text_inputs = {key: val.squeeze(1).to(device) for key, val in text_inputs.items()}
            img_inputs = img_inputs.to(device)

            outputs = model(text_inputs, img_inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct_preds / total_preds
    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# 预测测试数据
def predict_testset(model, test_loader):
    predictions = []
    with torch.no_grad():
        for text_inputs, img_inputs, guids in test_loader:
            text_inputs = {key: val.squeeze(1).to(device) for key, val in text_inputs.items()}
            img_inputs = img_inputs.to(device)

            # 获取模型输出
            outputs = model(text_inputs, img_inputs)
            _, predicted = torch.max(outputs, 1)

            # 反向映射标签
            predicted_labels = [reverse_label_map[label.item()] for label in predicted]

            for guid, label in zip(guids, predicted_labels):
                predictions.append((guid, label))

    return predictions

# 保存预测结果到文件
def save_predictions(predictions, output_file):
    df = pd.read_csv(output_file)
    df['tag'] = df['tag'].astype('object')
    df['guid'] = df['guid'].astype(int)
    for idx, (guid, label) in enumerate(predictions):
        guid = int(guid)
        df.loc[df['guid'] == guid, 'tag'] = label
    df.to_csv(output_file, index=False)
    print(f"预测结果已经保存到 {output_file} 文件")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
resnet_model = models.resnet18(pretrained=True).to(device)
# 去掉最后的分类层
resnet_model.fc = nn.Identity()