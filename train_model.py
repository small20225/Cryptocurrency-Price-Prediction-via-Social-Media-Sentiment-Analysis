import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np

# 下载必要的NLTK资源
nltk.download('vader_lexicon')

# 读取数据
data = pd.read_csv('GPTtestData_標記.csv')

# 文本预处理函数
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())  # 去除标点符号并转换为小写
    return text

# 应用预处理函数
data['Content'] = data['Content'].apply(preprocess_text)

# 情感分析
sid = SentimentIntensityAnalyzer()
data['sentiment'] = data['Content'].apply(lambda x: sid.polarity_scores(x)['compound'])

# 其他文本特征
data['word_count'] = data['Content'].apply(lambda x: len(str(x).split()))
data['unique_word_ratio'] = data['Content'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0)

# TF-IDF 特征提取
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
tfidf_features = tfidf.fit_transform(data['Content']).toarray()

# 结合所有特征
features = np.hstack((tfidf_features, data[['sentiment', 'word_count', 'unique_word_ratio']].values))

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 选择目标变量并填补缺失值
labels_change_percentage = data['Change_Percentage'].fillna(0).values

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_change_percentage, test_size=0.2, random_state=42)

# 转换数据格式为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定义简单的神经网络模型（DNN）
class DNNModel(nn.Module):
    def __init__(self, input_dim):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 初始化DNN模型
input_dim = X_train.shape[1]
model_dnn = DNNModel(input_dim)
optimizer_dnn = optim.Adam(model_dnn.parameters(), lr=0.001)

# 训练函数
def train_model(model, optimizer, X_train, y_train, X_test, y_test, epochs=75):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

# 训练DNN模型
train_model(model_dnn, optimizer_dnn, X_train, y_train, X_test, y_test)

# 保存模型
torch.save(model_dnn.state_dict(), 'dnn_model_change_percentage.pth')
torch.save(scaler, 'scaler.pth')
torch.save(tfidf, 'tfidf.pth')
torch.save(sid, 'sid.pth')
