import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
from torchvision import transforms
from transformers import BertModel
from torchvision import models
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from timm import create_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.data_model_utils import MultimodalDataset,get_text_features,train,evaluate,predict_testset,save_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 获取图像特征
# def get_image_features(image):
#     with torch.no_grad():
#         image = image.to(device)  # 将图像移到设备上
#         features = vit_model(image)  # 获取中间层特征
#         features = features[:, 0, :]
#     return features

# 跨模态注意力模块
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)

    def forward(self, text_features, image_features):
        # 输入(batch_size,hidden_size)，输出(1,batch_size,hidden_size)
        text_features = text_features.unsqueeze(0)
        image_features = image_features.unsqueeze(0)

        attn_output, attn_output_weights = self.multihead_attention(text_features, image_features, image_features)
        # 返回(batch_size,hidden_size)
        return attn_output.squeeze(0)

# 多模态模型
class MultimodalModel(nn.Module):
    def __init__(self, text_model, img_model, hidden_size, num_classes, transformer_layers=2, num_heads=8):
        super(MultimodalModel, self).__init__()

        self.text_model = text_model
        self.img_model= img_model
        # BERT输出768维的特征
        self.text_fc = nn.Linear(768, hidden_size)
        # ResNet输出512维的特征
        self.img_fc = nn.Linear(1000, hidden_size)
        self.cross_modal_attention = CrossModalAttention(hidden_size, num_heads)
        self.transformer = nn.Transformer(hidden_size, num_encoder_layers=transformer_layers)

        self.fc1 = nn.Linear(3 * hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, text_inputs, img_inputs):
        # text_features = get_text_features(encoding)
        text_features = self.text_model(**text_inputs).pooler_output
        # print(f"Text features shape: {text_features.shape}")
        image_features = self.img_model(img_inputs)
        # image_features = get_image_features(img_inputs)
        # print(f"Image features shape: {image_features.shape}")

        text_features = self.text_fc(text_features)
        image_features = self.img_fc(image_features)

        # 通过跨模态注意力融合文本和图像特征
        attention_output = self.cross_modal_attention(text_features, image_features)

        combined_features = torch.cat((attention_output, text_features, image_features), dim=1)
        combined_features = self.fc1(combined_features)
        combined_features = combined_features.unsqueeze(0)  # (1,batch_size,3*hidden_size)
        transformer_out = self.transformer(combined_features,combined_features)
        transformer_out = transformer_out.squeeze(0)  # (batch_size,3*hidden_size)

        output = F.relu(self.fc2(transformer_out))
        output = self.fc3(output)

        return output

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
vit_model = create_model('vit_base_patch16_224', pretrained=True)
vit_model.fc = nn.Identity()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = 'dataset/data'
train_file = 'dataset/train.txt'
test_file = 'dataset/test_without_label.txt'
batch_size = 16

dataset = MultimodalDataset(data_dir, train_file, tokenizer, transform)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = MultimodalDataset(data_dir, test_file, tokenizer, transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print(f"训练集批次数量: {len(train_loader)}")
# print(f"验证集批次数量: {len(val_loader)}")
# print(f"测试集批次数量: {len(test_loader)}")
# print(f"训练数据集总样本数: {len(train_loader.dataset)}")
# print(f"验证数据集总样本数: {len(val_loader.dataset)}")
# print(f"测试数据集总样本数: {len(test_loader.dataset)}")

model = MultimodalModel(bert_model, vit_model, hidden_size=512, num_classes=3).to(device)
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

# 学习率调度器
step_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_model = None
best_val_loss = float('inf')
patience = 3
counter = 0
num_epochs=100

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_loss, val_accuracy = evaluate(model, val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

    step_scheduler.step()
    plateau_scheduler.step(val_loss)

torch.save(best_model, "output/best_model.pth")

# 绘制损失下降图
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig("output/loss_curve.png")
plt.show()
plt.close()

print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")

# 加载保存的最佳模型
model = MultimodalModel(bert_model, vit_model, hidden_size=512, num_classes=3).to(device)
model.load_state_dict(torch.load("output/best_model.pth"))
model.eval()
# 对预测集进行预测
predictions = predict_testset(model, test_loader)
save_predictions(predictions,test_file)
