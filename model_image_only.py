from torch import nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
from torchvision import transforms
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from utils.data_model_utils import MultimodalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ImageOnlyModel(nn.Module):
    def __init__(self, img_model, hidden_size, num_classes):
        super(ImageOnlyModel, self).__init__()
        self.img_model = img_model
        self.img_fc = nn.Linear(512, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, img_inputs):
        image_features = self.img_model(img_inputs)
        image_features = self.img_fc(image_features)
        output = self.fc(image_features)
        return output

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for _, img_inputs, labels in train_loader:
        labels = labels.to(device)
        img_inputs = img_inputs.to(device)
        optimizer.zero_grad()
        outputs = model(img_inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100 * correct_preds / total_preds
    print(f'Epoch [{epoch + 1}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# 验证函数
def evaluate(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for _, img_inputs, labels in val_loader:
            labels = labels.to(device)
            img_inputs = img_inputs.to(device)
            outputs = model(img_inputs)
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
resnet_model = models.resnet18(pretrained=True).to(device)
# 去掉最后的分类层
resnet_model.fc = nn.Identity()

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

model = ImageOnlyModel(resnet_model, hidden_size=512, num_classes=3).to(device)

# 定义损失函数和优化器
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
num_epochs=500

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

# 绘制损失下降图
plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig("loss_curve.png")
plt.show()
plt.close()

print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
