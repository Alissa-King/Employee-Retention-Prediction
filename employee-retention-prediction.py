import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv')

df

# Data cleaning and preprocessing
def preprocess_data(df):
    # Separate the 'Attrition' column
    df['Attrition'] = df['Attrition'].astype('category')
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Convert categorical variables to numeric
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Drop irrelevant columns if they exist
    columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df

df_processed = preprocess_data(df)

# Split the data into features and target
X = df_processed.drop('Attrition', axis=1)
y = df_processed['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Exploratory Data Analysis
def plot_attrition_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Attrition', data=df)
    plt.title('Distribution of Attrition')
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_feature_importance(df):
    correlation = df.corr()['Attrition'].sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlation.values[1:11], y=correlation.index[1:11])
    plt.title('Top 10 Features Correlated with Attrition')
    plt.show()

# Run EDA functions
plot_attrition_distribution(df_processed)

plot_correlation_heatmap(df_processed)

plot_feature_importance(df_processed)

print("Data preprocessing and EDA completed.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize models
logistic = LogisticRegression(random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
xgboost = XGBClassifier(n_estimators=100, random_state=42)

# Train models
logistic.fit(X_train_scaled, y_train)
random_forest.fit(X_train, y_train)
xgboost.fit(X_train, y_train)

# Make predictions
y_pred_logistic = logistic.predict(X_test_scaled)
y_pred_rf = random_forest.predict(X_test)
y_pred_xgb = xgboost.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

evaluate_model(y_test, y_pred_logistic, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Feature importance analysis
def plot_feature_importance(model, X, model_name):
    if model_name == "Logistic Regression":
        importance = abs(model.coef_[0])
    else:
        importance = model.feature_importances_
    
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Top 10 Feature Importance - {model_name}')
    plt.show()

plot_feature_importance(logistic, X, "Logistic Regression")

plot_feature_importance(random_forest, X, "Random Forest")

plot_feature_importance(xgboost, X, "XGBoost")

# Model interpretation using SHAP values for XGBoost
import shap

explainer = shap.TreeExplainer(xgboost)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar")

print("Model training, evaluation, and interpretation completed.")
