import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATASET
# The dataset uses ';' as a delimiter
# Use this if you move the csv into a 'data' folder
df = pd.read_csv('cardio_train.csv', sep=';')

# 2. DATA PRE-PROCESSING
# Drop the ID column as it doesn't contribute to prediction
df.drop('id', axis=1, inplace=True)

# Convert age from days to years
df['age'] = (df['age'] / 365).astype(int)

# Handling Outliers in Blood Pressure (ap_hi and ap_lo)
# Removing unrealistic values and ensuring systolic > 
# Removing outliers: Blood pressure cannot be negative or realistically above 250/150
df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
df = df[df['ap_hi'] > df['ap_lo']]

# Handling Height and Weight Outliers
df = df[(df['height'] >= 140) & (df['height'] <= 200)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 150)]

# Remove duplicate entries
df.drop_duplicates(inplace=True)

print(f"Data cleaning complete. Remaining samples: {df.shape[0]}")

# 3. DATA ANALYSIS & VISUALIZATIONS
# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Age (Years)')
plt.savefig('age_distribution.png')

# Impact of Cholesterol and Glucose on Cardio
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x='cholesterol', hue='cardio', data=df, ax=axes[0], palette='Set2')
axes[0].set_title('Cholesterol levels vs Disease')
sns.countplot(x='gluc', hue='cardio', data=df, ax=axes[1], palette='Set1')
axes[1].set_title('Glucose levels vs Disease')
plt.savefig('categorical_analysis.png')

# 4. CORRELATION MATRIX
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_heatmap.png')

# 5. MACHINE LEARNING MODELING
# Define Features (X) and Target (y)
X = df.drop('cardio', axis=1)
y = df['cardio']

# Feature Scaling (Important for SVM, KNN, and Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf') # Note: SVM may take time on large datasets
}

accuracy_results = {}

print("\n--- Model Accuracy Levels ---")
for name, model in models.items():
    # If dataset is very large, SVM/KNN training can be slow. 
    # For this demonstration, we train on the full processed set.
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    accuracy_results[name] = acc
    print(f"{name}: {acc*100:.2f}%")

# 6. FINAL MODEL COMPARISON PLOT
plt.figure(figsize=(10, 6))
res_df = pd.DataFrame(list(accuracy_results.items()), columns=['Model', 'Accuracy'])
sns.barplot(x='Accuracy', y='Model', data=res_df.sort_values(by='Accuracy', ascending=False))
plt.title('Comparison of Model Accuracies')
plt.savefig('model_comparison.png')

# Identify the best model
best_model = res_df.sort_values(by='Accuracy', ascending=False).iloc[0]
print(f"\nBest Model: {best_model['Model']} with {best_model['Accuracy']*100:.2f}% accuracy.")
# Updated Prediction Code (Fixes the Warning)
feature_names = X.columns # This gets ['age', 'gender', 'height', 'weight', ...]

# Create a DataFrame for the new patient to include feature names
new_patient_data = pd.DataFrame([[50, 2, 170, 80, 140, 90, 2, 1, 0, 0, 1]], 
                                columns=feature_names)

# Scale and Predict
new_patient_scaled = scaler.transform(new_patient_data)
prediction = models["SVM"].predict(new_patient_scaled)

if prediction[0] == 1:
    print("Prediction: High Risk of Cardiovascular Disease")
else:
    print("Prediction: Low Risk of Cardiovascular Disease")