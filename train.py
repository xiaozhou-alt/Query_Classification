import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd

# 设置随机种子确保可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集类
class MedicalQueryDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query1 = str(item['query1'])
        query2 = str(item['query2'])
        label = int(item['label']) if item['label'] != '' else -1
        
        encoding = self.tokenizer.encode_plus(
            query1,
            query2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 训练函数（添加早停机制）
def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, model_save_path):
    best_accuracy = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    # 早停参数
    patience = 5  # 连续5个epoch验证集性能没有提升则停止
    min_delta = 0.001  # 认为有提升的最小变化量
    no_improve_count = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        total_steps = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
            total_steps += 1
        
        if total_steps > 0:
            avg_train_loss = train_loss / total_steps
        else:
            avg_train_loss = float('nan')
        
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        val_loss, val_accuracy = evaluate_model(model, val_loader)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
        
        # 检查是否达到最佳准确率
        if val_accuracy > best_accuracy + min_delta:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with accuracy: {best_accuracy:.4f}')
            no_improve_count = 0  # 重置计数器
        else:
            no_improve_count += 1
            print(f'No improvement for {no_improve_count} epochs. Best accuracy: {best_accuracy:.4f}')
        
        # 早停检查
        if no_improve_count >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs!')
            break
    
    return history

# 评估函数
def evaluate_model(model, data_loader, sample_size=10):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            val_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            total_steps += 1
    
    # 计算整体指标
    if total_steps > 0:
        avg_val_loss = val_loss / total_steps
    else:
        avg_val_loss = float('nan')
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 随机抽样展示
    print("\nSample Validation Predictions:")
    sample_indices = random.sample(range(len(all_labels)), min(sample_size, len(all_labels)))
    
    for i in sample_indices:
        actual_label = all_labels[i]
        pred_label = all_preds[i]
        print(f"Sample {i+1}: Actual={actual_label}, Predicted={pred_label} - {'✓' if actual_label == pred_label else '✗'}")
    
    return avg_val_loss, accuracy

# 保存训练历史为图片
def save_training_history(history, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据框
    df = pd.DataFrame({
        'epoch': range(1, len(history['train_loss']) + 1),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    })
    
    # 保存为CSV
    csv_path = os.path.join(output_dir, 'training_history.csv')
    df.to_csv(csv_path, index=False)
    print(f"Training history saved to {csv_path}")
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    
    # 训练和验证损失图
    plt.subplot(2, 1, 1)
    plt.plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
    plt.plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 验证准确率图
    plt.subplot(2, 1, 2)
    plt.plot(df['epoch'], df['val_acc'], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # 确保准确率在0-1之间
    plt.legend()
    plt.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    img_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(img_path)
    plt.close()
    
    print(f"Training history plot saved to {img_path}")
    
    return df

# 主函数
def main():
    # 配置参数
    MODEL_NAME = 'hfl/chinese-roberta-wwm-ext-large'
    MAX_LEN = 256
    BATCH_SIZE = 24
    EPOCHS = 30
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = '/kaggle/working/best_medical_query_model.bin'
    HISTORY_DIR = '/kaggle/working/training_history'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # 加载数据
    with open('/kaggle/input/query-classification/KUAKE-QQR_train.json', 'r') as f:
        train_data = json.load(f)
    
    with open('/kaggle/input/query-classification/KUAKE-QQR_dev.json', 'r') as f:
        val_data = json.load(f)
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")
    
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3  # 0, 1, 2三个类别
    )
    model = model.to(device)
    
    # 创建数据加载器
    train_dataset = MedicalQueryDataset(train_data, tokenizer, MAX_LEN)
    val_dataset = MedicalQueryDataset(val_data, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 训练模型
    print("Starting training...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        EPOCHS,
        MODEL_SAVE_PATH
    )
    
    # 保存训练历史
    save_training_history(history, HISTORY_DIR)
    
    # 在测试集上进行预测
    print("\nPredicting on test set...")
    with open('/kaggle/input/query-classification/KUAKE-QQR_test.json', 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # 加载最佳模型
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # 预测并保存结果
    output_path = '/kaggle/working/KUAKE-QQR_test_pred.json'
    with open(output_path, 'w') as f:
        for item in tqdm(test_data, desc="Predicting"):
            encoding = tokenizer.encode_plus(
                str(item['query1']),
                str(item['query2']),
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).cpu().item()
            
            item['label'] = str(pred)
        
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Test predictions saved to {output_path}")

if __name__ == '__main__':
    main()