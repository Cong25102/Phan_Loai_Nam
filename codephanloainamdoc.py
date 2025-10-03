# 1. Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Đọc dữ liệu
df = pd.read_csv('mushrooms.csv')  # Đảm bảo bạn có file này trong thư mục làm việc

# 3. Kiểm tra cấu trúc
print("Kích thước dữ liệu:", df.shape)
print("\nThông tin dữ liệu:")
print(df.info())

# 4. Phân tích biến mục tiêu (cột 'class': edible=e, poisonous=p)
plt.figure(figsize=(6,4))
sns.countplot(x='class', data=df, palette='Set2')
plt.title('Phân phối biến mục tiêu (class)')
plt.xlabel('Loại nấm')
plt.ylabel('Số lượng')
plt.show()

# 5. Phân tích đặc trưng phân loại (vì toàn bộ dữ liệu là categorical)
for col in df.columns:
    if col != 'class':
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette='Set3')
        plt.title(f'Tần suất giá trị của đặc trưng: {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 6. Kiểm tra dữ liệu thiếu
missing = df.isnull().sum()
print("\nSố lượng giá trị thiếu mỗi cột:")
print(missing[missing > 0])

# 7. Mã hóa dữ liệu để tính ma trận tương quan
df_encoded = df.apply(lambda x: pd.factorize(x)[0])  # Chuyển categorical thành số nguyên

# 8. Ma trận tương quan
plt.figure(figsize=(12,10))
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Ma trận tương quan giữa các đặc trưng')
plt.show()

# Thay thế dữ liệu thiếu bằng mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)



from sklearn.preprocessing import OneHotEncoder

df_encoded = pd.get_dummies(df.drop('class', axis=1))  # loại bỏ biến mục tiêu

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

df['odor_spore'] = df['odor'] + "_" + df['spore-print-color']

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = df_encoded.shape[1]
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(df_encoded, df_encoded, epochs=50, batch_size=32, validation_split=0.2)

features_autoencoded = encoder.predict(df_encoded)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Đọc dữ liệu
df = pd.read_csv('mushrooms.csv')

# Biến mục tiêu
y = df['class'].map({'e': 0, 'p': 1})  # edible=0, poisonous=1

# Xử lý dữ liệu thiếu
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# One-Hot Encoding cho đặc trưng
X_encoded = pd.get_dummies(df.drop('class', axis=1))

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Tách dữ liệu
X_train_A, X_test_A, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model_A = RandomForestClassifier(random_state=42)
model_A.fit(X_train_A, y_train)

# Dự đoán và đánh giá
y_pred_A = model_A.predict(X_test_A)
print("🔍 Kết quả với Luồng A (Truyền thống):")
print("Accuracy:", accuracy_score(y_test, y_pred_A))
print(classification_report(y_test, y_pred_A))

# Kiến trúc Autoencoder
input_dim = X_encoded.shape[1]
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

# Huấn luyện Autoencoder
autoencoder.compile(optimizer=Adam(), loss='mse')
autoencoder.fit(X_encoded, X_encoded, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Trích xuất đặc trưng
X_autoencoded = encoder.predict(X_encoded)

# Tách dữ liệu
X_train_B, X_test_B, _, _ = train_test_split(X_autoencoded, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model_B = RandomForestClassifier(random_state=42)
model_B.fit(X_train_B, y_train)

# Dự đoán và đánh giá
y_pred_B = model_B.predict(X_test_B)
print("🔍 Kết quả với Luồng B (Autoencoder):")
print("Accuracy:", accuracy_score(y_test, y_pred_B))
print(classification_report(y_test, y_pred_B))

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Bộ dữ liệu
X_A = X_scaled
X_B = features_autoencoded

# Tách dữ liệu
X_train_A, X_test_A, y_train, y_test = train_test_split(X_A, y, test_size=0.2, random_state=42)
X_train_B, X_test_B, _, _ = train_test_split(X_B, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Huấn luyện và đánh giá
results = []

for name, model in models.items():
    # Luồng A
    model.fit(X_train_A, y_train)
    y_pred_A = model.predict(X_test_A)
    results.append({
        "Model": name,
        "Feature Set": "Luồng A",
        "Accuracy": accuracy_score(y_test, y_pred_A),
        "Precision": precision_score(y_test, y_pred_A),
        "Recall": recall_score(y_test, y_pred_A),
        "F1-score": f1_score(y_test, y_pred_A),
        "ROC-AUC": roc_auc_score(y_test, y_pred_A)
    })

    # Luồng B
    model.fit(X_train_B, y_train)
    y_pred_B = model.predict(X_test_B)
    results.append({
        "Model": name,
        "Feature Set": "Luồng B",
        "Accuracy": accuracy_score(y_test, y_pred_B),
        "Precision": precision_score(y_test, y_pred_B),
        "Recall": recall_score(y_test, y_pred_B),
        "F1-score": f1_score(y_test, y_pred_B),
        "ROC-AUC": roc_auc_score(y_test, y_pred_B)
    })

# Chuyển kết quả thành bảng
results_df = pd.DataFrame(results)
print("📊 Bảng so sánh hiệu suất mô hình:")
print(results_df)
