
# CREDIT CARD FRAUD DETECTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# Load dataset
df = pd.read_csv("creditcard.csv")
print("First few rows:")
print(df.head())


# Checking missing values
print("\nMissing values:", df.isnull().sum().sum())
print("\nClass count before balancing:")
print(df["Class"].value_counts())

# Separate features and target
features = df.drop("Class", axis=1)
labels = df["Class"]

# Standardizing numerical columns
scale = StandardScaler()
features[["Amount", "Time"]] = scale.fit_transform(features[["Amount", "Time"]])

# Oversampling using SMOTE (class imbalance)
oversample = SMOTE(random_state=1, k_neighbors=2)
X_balanced, y_balanced = oversample.fit_resample(features, labels)

print("\nBalanced dataset distribution:")
print(Counter(y_balanced))


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=1, stratify=y_balanced
)
print(f"\nTrain data: {len(X_train)}, Test data: {len(X_test)}")


# Training different models
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=120, random_state=1)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    n_estimators=120,
    learning_rate=0.08,
    max_depth=6,
    random_state=1,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)


# Evaluation function
def evaluate(title, model):
    preds = model.predict(X_test)
    
    print("\n",title,"\n" )
   
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
    plt.title(title + " (Confusion Matrix)")
    plt.show()

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("ROC-AUC:", round(auc, 4))


# Evaluate all models
evaluate("Logistic Regression", log_model)
evaluate("Random Forest", rf_model)
evaluate("XGBoost", xgb_model)


# Precision-Recall curve for Random Forest
probs = rf_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.title("Precision vs Recall (Random Forest)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()


# Show feature importance (RF)
importance = pd.Series(rf_model.feature_importances_, index=features.columns)
importance.sort_values(ascending=False).head(12).plot(kind="barh", figsize=(8, 5))
plt.title("Important Features (Random Forest)")
plt.show()
