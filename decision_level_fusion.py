import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils.data_model_utils import MultimodalDataset,train,evaluate,predict_testset,save_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 文本模型输出的信心值
def get_text_logits(encoding):
    input_ids = encoding['input_ids'].squeeze(1).to(device)
    attention_mask = encoding['attention_mask'].squeeze(1).to(device)
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    return logits

# 图片模型输出的信心值
def get_image_logits(image):
    with torch.no_grad():
        image = image.to(device)
        outputs = resnet_model(image)
        logits = outputs
    return logits

# 多模态模型(决策级融合)
class MultimodalModel(nn.Module):
    def __init__(self, text_model, img_model, hidden_size, num_classes):
        super(MultimodalModel, self).__init__()
        self.bert_model = text_model
        self.resnet_model = img_model
        self.fc_image = nn.Linear(1000, num_classes)

    def forward(self, text_inputs, img_inputs):
        text_logits = get_text_logits(text_inputs)
        image_logits = get_image_logits(img_inputs)
        image_logits = self.fc_image(image_logits)

        combined_logits = (text_logits + image_logits) / 2

        return combined_logits

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)
resnet_model = models.resnet18(pretrained=True).to(device)

# 图像的变换
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

model = MultimodalModel(bert_model, resnet_model, hidden_size=512, num_classes=3).to(device)
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

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
model = MultimodalModel(bert_model, resnet_model, hidden_size=512, num_classes=3).to(device)
model.load_state_dict(torch.load("output/best_model.pth"))
model.eval()

# 对预测集进行预测
predictions = predict_testset(model, test_loader)
save_predictions(predictions,test_file)