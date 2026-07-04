import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# CSV dosyasını yükle
df = pd.read_csv("heart.csv")

print("Veri seti boyutu:", df.shape)
print("\nİlk 5 satır:")
print(df.head())

print("\nEksik değerler:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

# Tekrar eden satırları sil
df.drop_duplicates(inplace=True)
print("\nTemizleme sonrası veri seti boyutu:", df.shape)

# Hedef sütun (target) (1: Kalp hastalığı var, 0: Yok)
X = df.drop("target", axis=1)
y = df["target"]

print("\nSınıf dağılımı:")
print(y.value_counts())

# Veriyi %80 eğitim, %20 test olarak böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling uygula
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                 SVC(kernel="rbf", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42)
}

for model_name, model in models.items():

    # Modeli eğitim verisiyle eğit
    model.fit(X_train, y_train)

    # Test verisiyle tahmin yap
    y_pred = model.predict(X_test)

    # Accuracy Score hesapla
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion Matrix hesapla
    cm = confusion_matrix(y_test, y_pred)

    # Sonuçları yazdır
    print(f"\n{'-'*40}")
    print(f"Model: {model_name}")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"Confusion Matrix:")
    print(cm)

'''
SONUÇ
    Heart Disease veri seti 13 özellik içeren dengeli bir sınıflandırma veri setidir.
    Duplicate temizleme sonrası 1025 satırdan 302 satıra düşmüştür.
    Random Forest en yüksek doğruluk oranını (%83.61) elde etmiştir çünkü
    özellikler arasındaki karmaşık ilişkileri ensemble yöntemiyle daha iyi yakalayabilir.
    k-NN ise komşuluk tabanlı yapısı nedeniyle bu veri setinde en düşük performansı
    göstermiştir.
'''