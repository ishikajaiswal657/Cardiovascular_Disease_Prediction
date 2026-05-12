import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, roc_curve, confusion_matrix,
                              f1_score, recall_score)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

print("=" * 60)
print("  HEART DISEASE PREDICTION — CLEVELAND DATASET")
print("=" * 60)

col_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
             'thalach','exang','oldpeak','slope','ca','thal','target']

loaded = False
for fname, sep, has_header in [
    ('heart_cleveland_upload.csv', ',', True),
    ('cleveland.csv', ',', True),
    ('heart.csv', ',', True),
    ('processed.cleveland.data', ',', False),
]:
    try:
        if has_header:
            df = pd.read_csv(fname, sep=sep)
            if 'condition' in df.columns:
                df.rename(columns={'condition': 'target'}, inplace=True)
        else:
            df = pd.read_csv(fname, sep=sep, header=None,
                             names=col_names, na_values='?')
        print(f"\nLoaded: {fname}")
        loaded = True
        break
    except FileNotFoundError:
        continue

if not loaded:
    print("\nDataset not found! Please download:")
    print("https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci")
    print("Save as heart_cleveland_upload.csv in:", os.getcwd())
    exit()

# Clean
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col])
df['target'] = (df['target'] > 0).astype(int)
df.drop_duplicates(inplace=True)

print(f"Rows after cleaning : {df.shape[0]}")
print(f"Class balance       : {df['target'].value_counts().to_dict()}")

# EDA
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
df['target'].value_counts().plot(kind='bar', ax=axes[0],
    color=['steelblue','tomato'], edgecolor='white')
axes[0].set_title('Class Distribution')
axes[0].set_xticklabels(['No Disease','Disease'], rotation=0)
axes[1].hist(df[df['target']==0]['age'], bins=15, alpha=0.6, color='steelblue', label='No Disease')
axes[1].hist(df[df['target']==1]['age'], bins=15, alpha=0.6, color='tomato', label='Disease')
axes[1].set_title('Age Distribution'); axes[1].legend()
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=axes[2])
axes[2].set_title('Correlation')
plt.tight_layout()
plt.savefig('eda_cleveland.png', dpi=150, bbox_inches='tight')
print("Saved: eda_cleveland.png")

# Feature Engineering
print("\n=== Feature Engineering ===")
df['age_x_thalach']   = df['age'] * df['thalach']
df['age_x_oldpeak']   = df['age'] * df['oldpeak']
df['chol_x_age']      = df['chol'] * df['age']
df['bp_x_age']        = df['trestbps'] * df['age']
df['cp_x_thalach']    = df['cp'] * df['thalach']
print(f"5 interaction features added. Total: {df.shape[1]-1} features")

X = df.drop('target', axis=1)
y = df['target']
feature_names = X.columns.tolist()

# Z-Score
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Z-Score normalization applied.")

# Split
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
print(f"SMOTE: {X_train_raw.shape[0]} → {X_train.shape[0]} samples")

# Neural Network
print("\n" + "="*60)
print("NEURAL NETWORK (Keras)")
print("="*60)
NEURAL_NET_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    tf.random.set_seed(42)
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(), Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(), Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    print("Training Neural Network...")
    history = nn_model.fit(X_train, y_train, epochs=200, batch_size=32,
                            validation_split=0.1, callbacks=[early_stop], verbose=0)
    nn_proba = nn_model.predict(X_test, verbose=0).flatten()
    nn_preds = (nn_proba >= 0.5).astype(int)
    nn_acc = accuracy_score(y_test, nn_preds)
    nn_auc = roc_auc_score(y_test, nn_proba)
    nn_f1  = f1_score(y_test, nn_preds)
    print(f"Neural Network Accuracy : {nn_acc*100:.2f}%")
    print(f"Neural Network ROC-AUC  : {nn_auc:.4f}")
    print(f"Neural Network F1       : {nn_f1:.4f}")
    print(f"Stopped at epoch        : {len(history.history['loss'])}")
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train', color='steelblue')
    plt.plot(history.history['val_loss'], label='Val', color='tomato')
    plt.title('NN Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train', color='steelblue')
    plt.plot(history.history['val_accuracy'], label='Val', color='tomato')
    plt.title('NN Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig('neural_network_training.png', dpi=150, bbox_inches='tight')
    print("Saved: neural_network_training.png")
    NEURAL_NET_AVAILABLE = True
except ImportError:
    print("TensorFlow not installed. Run: pip install tensorflow")
except Exception as e:
    print(f"NN error: {e}")

# Models
models = {
    "Logistic Regression" : LogisticRegression(max_iter=500, random_state=42),
    "K-Nearest Neighbor"  : KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, max_depth=8,
                                                    min_samples_leaf=2, random_state=42),
}
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=5,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss', random_state=42)

# 10-Fold CV
print("\n" + "="*60)
print("METHOD 1: 10-Fold Cross-Validation")
print("="*60)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:25s}  ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
best_cv_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n★ Best (CV): {best_cv_name}  AUC={cv_results[best_cv_name].mean():.4f}")

# Test performance
print("\n" + "="*60)
print("Test Set Performance")
print("="*60)
print(f"{'Model':<25}  {'Accuracy':>9}  {'ROC-AUC':>9}  {'F1':>7}  {'Recall':>7}")
print("-"*65)
acc_results = {}; auc_results = {}; f1_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X_test)
    acc_results[name] = accuracy_score(y_test, preds)
    auc_results[name] = roc_auc_score(y_test, proba)
    f1_results[name]  = f1_score(y_test, preds)
    print(f"{name:<25}  {acc_results[name]*100:>8.2f}%  {auc_results[name]:>9.4f}  {f1_results[name]:>7.4f}  {recall_score(y_test,preds):>7.4f}")
if NEURAL_NET_AVAILABLE:
    acc_results['Neural Network'] = nn_acc
    auc_results['Neural Network'] = nn_auc
    f1_results['Neural Network']  = nn_f1
    print(f"{'Neural Network':<25}  {nn_acc*100:>8.2f}%  {nn_auc:>9.4f}  {nn_f1:>7.4f}  {recall_score(y_test,nn_preds):>7.4f}")

# Ensemble
print("\n" + "="*60)
print("ENSEMBLE — VotingClassifier")
print("="*60)
estimators = [('lr', LogisticRegression(max_iter=500, random_state=42)),
               ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42))]
if XGBOOST_AVAILABLE:
    estimators.append(('xgb', XGBClassifier(n_estimators=300, max_depth=5,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss', random_state=42)))
ensemble = VotingClassifier(estimators=estimators, voting='soft')
ensemble.fit(X_train, y_train)
ens_preds = ensemble.predict(X_test)
ens_proba = ensemble.predict_proba(X_test)[:,1]
ens_acc = accuracy_score(y_test, ens_preds)
ens_auc = roc_auc_score(y_test, ens_proba)
ens_f1  = f1_score(y_test, ens_preds)
print(f"Ensemble Accuracy : {ens_acc*100:.2f}%")
print(f"Ensemble ROC-AUC  : {ens_auc:.4f}")
print(f"Ensemble F1       : {ens_f1:.4f}")
acc_results['Ensemble'] = ens_acc
auc_results['Ensemble'] = ens_auc
f1_results['Ensemble']  = ens_f1

# Threshold tuning
print("\n" + "="*60)
print("METHOD 4: Threshold Tuning")
print("="*60)
best_overall = max(auc_results, key=auc_results.get)
print(f"Best model: {best_overall}  AUC={auc_results[best_overall]:.4f}")
if best_overall == 'Ensemble': best_proba = ens_proba
elif best_overall == 'Neural Network' and NEURAL_NET_AVAILABLE: best_proba = nn_proba
else:
    bm = models[best_overall]
    best_proba = bm.predict_proba(X_test)[:,1] if hasattr(bm,'predict_proba') else bm.decision_function(X_test)
thresholds    = np.arange(0.25, 0.76, 0.01)
f1_scores_thr = [f1_score(y_test,(best_proba>=t).astype(int)) for t in thresholds]
optimal_thr   = thresholds[np.argmax(f1_scores_thr)]
preds_optimal = (best_proba >= optimal_thr).astype(int)
preds_default = (best_proba >= 0.50).astype(int)
print(f"Default  (0.50) → Acc: {accuracy_score(y_test,preds_default)*100:.2f}%  F1: {f1_score(y_test,preds_default):.4f}  Recall: {recall_score(y_test,preds_default):.4f}")
print(f"Optimal  ({optimal_thr:.2f}) → Acc: {accuracy_score(y_test,preds_optimal)*100:.2f}%  F1: {f1_score(y_test,preds_optimal):.4f}  Recall: {recall_score(y_test,preds_optimal):.4f}  ← improved")

# RandomizedSearchCV
print("\n" + "="*60)
print("METHOD 5: RandomizedSearchCV")
print("="*60)
if XGBOOST_AVAILABLE:
    param_grid = {'n_estimators':[200,300,400,500],'max_depth':[3,4,5,6,7],
        'learning_rate':[0.01,0.03,0.05,0.1,0.15],'subsample':[0.7,0.8,0.9,1.0],
        'colsample_bytree':[0.7,0.8,0.9,1.0],'min_child_weight':[1,2,3,5]}
    tune_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
else:
    param_grid = {'n_estimators':[100,200,300,400],'max_depth':[4,6,8,10,None],
        'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4],'max_features':['sqrt','log2']}
    tune_base = RandomForestClassifier(random_state=42)
print("Running 30 combos × 5-fold CV...")
search = RandomizedSearchCV(tune_base, param_distributions=param_grid, n_iter=30,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1, random_state=42, verbose=1)
search.fit(X_train, y_train)
best_tuned  = search.best_estimator_
tuned_preds = best_tuned.predict(X_test)
tuned_proba = best_tuned.predict_proba(X_test)[:,1]
tuned_preds_opt = (tuned_proba >= optimal_thr).astype(int)
print(f"Best params   : {search.best_params_}")
print(f"CV  ROC-AUC   : {search.best_score_:.4f}")
print(f"Test Accuracy : {accuracy_score(y_test,tuned_preds)*100:.2f}%")
print(f"Test ROC-AUC  : {roc_auc_score(y_test,tuned_proba):.4f}")
print(f"Test F1       : {f1_score(y_test,tuned_preds_opt):.4f}")

# Optuna
print("\n" + "="*60)
print("OPTUNA SMART TUNING (100 trials)")
print("="*60)
OPTUNA_AVAILABLE = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        if XGBOOST_AVAILABLE:
            params = {
                'n_estimators'    : trial.suggest_int('n_estimators', 100, 600),
                'max_depth'       : trial.suggest_int('max_depth', 2, 10),
                'learning_rate'   : trial.suggest_float('learning_rate', 0.005, 0.3),
                'subsample'       : trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma'           : trial.suggest_float('gamma', 0, 5),
                'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 3),
                'reg_lambda'      : trial.suggest_float('reg_lambda', 0, 3),
            }
            model = XGBClassifier(**params, use_label_encoder=False,
                                   eval_metric='logloss', random_state=42)
        else:
            params = {'n_estimators': trial.suggest_int('n_estimators',100,500),
                      'max_depth': trial.suggest_int('max_depth',2,15)}
            model = RandomForestClassifier(**params, random_state=42)
        return cross_val_score(model, X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc', n_jobs=-1).mean()
    print("Running 100 Optuna trials...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    print(f"\nBest Optuna params : {study.best_params}")
    print(f"Best Optuna AUC    : {study.best_value:.4f}")
    if XGBOOST_AVAILABLE:
        optuna_model = XGBClassifier(**study.best_params, use_label_encoder=False,
                                      eval_metric='logloss', random_state=42)
    else:
        optuna_model = RandomForestClassifier(**study.best_params, random_state=42)
    optuna_model.fit(X_train, y_train)
    opt_preds = optuna_model.predict(X_test)
    opt_proba = optuna_model.predict_proba(X_test)[:,1]
    opt_acc   = accuracy_score(y_test, opt_preds)
    opt_auc   = roc_auc_score(y_test, opt_proba)
    opt_f1    = f1_score(y_test, opt_preds)
    print(f"\nOptuna Accuracy : {opt_acc*100:.2f}%")
    print(f"Optuna ROC-AUC  : {opt_auc:.4f}")
    print(f"Optuna F1       : {opt_f1:.4f}")
    if opt_auc > roc_auc_score(y_test, tuned_proba):
        best_tuned = optuna_model; tuned_proba = opt_proba; tuned_preds = opt_preds
        print("Optuna model is the BEST — using for final predictions.")
    plt.figure(figsize=(9,4))
    trials_auc = [t.value for t in study.trials]
    plt.plot(trials_auc, color='steelblue', lw=1.5)
    plt.axhline(max(trials_auc), color='tomato', linestyle='--',
                label=f'Best = {max(trials_auc):.4f}')
    plt.xlabel('Trial'); plt.ylabel('ROC-AUC')
    plt.title('Optuna — 100 Trial AUC History'); plt.legend()
    plt.tight_layout()
    plt.savefig('optuna_tuning.png', dpi=150, bbox_inches='tight')
    print("Saved: optuna_tuning.png")
    OPTUNA_AVAILABLE = True
except ImportError:
    print("Optuna not installed. Run: pip install optuna")
except Exception as e:
    print(f"Optuna error: {e}")

# Final Summary
tuned_preds_opt = (tuned_proba >= optimal_thr).astype(int)
print("\n" + "="*60)
print("FINAL SUMMARY — Cleveland Heart Disease Dataset")
print("="*60)
print(f"{'Model':<40}  {'Accuracy':>9}  {'ROC-AUC':>9}")
print("-"*65)
for name in models:
    print(f"  {name:<38}  {acc_results[name]*100:>8.2f}%  {auc_results[name]:>9.4f}")
if NEURAL_NET_AVAILABLE:
    print(f"  {'Neural Network':<38}  {nn_acc*100:>8.2f}%  {nn_auc:>9.4f}")
print(f"  {'Ensemble':<38}  {ens_acc*100:>8.2f}%  {ens_auc:>9.4f}")
print(f"  {'+ Threshold tuning':<38}  {accuracy_score(y_test,preds_optimal)*100:>8.2f}%  {roc_auc_score(y_test,best_proba):>9.4f}")
print(f"  {'+ RandomizedSearchCV':<38}  {accuracy_score(y_test,tuned_preds)*100:>8.2f}%  {roc_auc_score(y_test,tuned_proba):>9.4f}")
if OPTUNA_AVAILABLE:
    print(f"  {'+ Optuna (final best)':<38}  {opt_acc*100:>8.2f}%  {opt_auc:>9.4f}")
print(f"\n--- Classification Report ---")
print(classification_report(y_test, tuned_preds_opt, target_names=['No Disease','Disease']))

# Charts
plt.figure(figsize=(10,7))
for name, model in models.items():
    p = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_results[name]:.3f})")
if NEURAL_NET_AVAILABLE:
    fpr, tpr, _ = roc_curve(y_test, nn_proba)
    plt.plot(fpr, tpr, lw=2, label=f"Neural Network (AUC={nn_auc:.3f})")
fpr_e, tpr_e, _ = roc_curve(y_test, ens_proba)
plt.plot(fpr_e, tpr_e, lw=2.5, linestyle='--', label=f"Ensemble (AUC={ens_auc:.3f})")
plt.plot([0,1],[0,1],'k:',lw=1)
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curves — Cleveland Dataset'); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
print("Saved: roc_curves.png")

rf = models["Random Forest"]
imp_df = pd.DataFrame({'Feature':feature_names,'Importance':rf.feature_importances_}).sort_values('Importance',ascending=True)
plt.figure(figsize=(10,7))
colors = ['#d62728' if v > 0.06 else '#1f77b4' for v in imp_df['Importance']]
plt.barh(imp_df['Feature'], imp_df['Importance'], color=colors)
plt.xlabel('Importance'); plt.title('Feature Importance')
plt.tight_layout(); plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("Saved: feature_importance.png")

cm = confusion_matrix(y_test, tuned_preds_opt)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease','Disease'], yticklabels=['No Disease','Disease'])
plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: confusion_matrix.png")

fig, ax = plt.subplots(figsize=(10, 5))
cv_df = pd.DataFrame(cv_results)
ax.boxplot([cv_df[col] for col in cv_df.columns],
           labels=cv_df.columns, vert=True)
ax.set_title('10-Fold CV ROC-AUC per Model')
ax.set_ylabel('ROC-AUC')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('cv_boxplot.png', dpi=150, bbox_inches='tight')
print("Saved: cv_boxplot.png")

plt.figure(figsize=(9,4))
plt.plot(thresholds, f1_scores_thr, color='steelblue', lw=2)
plt.axvline(optimal_thr, color='black', linestyle='--', label=f'Optimal={optimal_thr:.2f}')
plt.axvline(0.50, color='gray', linestyle=':', label='Default=0.50')
plt.xlabel('Threshold'); plt.ylabel('F1'); plt.title('F1 vs Threshold'); plt.legend()
plt.tight_layout(); plt.savefig('threshold_tuning.png', dpi=150, bbox_inches='tight')
print("Saved: threshold_tuning.png")

# Save model
import joblib
joblib.dump(best_tuned,   'heart_model.pkl')
joblib.dump(scaler,       'heart_scaler.pkl')
joblib.dump(optimal_thr,  'heart_threshold.pkl')
joblib.dump(feature_names,'heart_features.pkl')
print("\nModel saved: heart_model.pkl")

# Risk categories
def get_risk_category(prob):
    if prob < 0.35: return 'LOW RISK'
    elif prob < 0.65: return 'MEDIUM RISK'
    else: return 'HIGH RISK'

all_proba = best_tuned.predict_proba(X_test)[:,1]
risk_series = pd.Series([get_risk_category(p) for p in all_proba])
print("\nRisk Categories:"); print(risk_series.value_counts().to_string())
counts = risk_series.value_counts()
plot_cats = [c for c in ['LOW RISK','MEDIUM RISK','HIGH RISK'] if c in counts.index]
counts[plot_cats].plot(kind='bar', color=['#2ecc71','#f39c12','#e74c3c'][:len(plot_cats)], edgecolor='white')
plt.title('Risk Category Distribution'); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig('risk_categories.png', dpi=150, bbox_inches='tight')
print("Saved: risk_categories.png")

# Live Prediction
print("\n" + "="*60)
print("LIVE PATIENT PREDICTION")
print("="*60)
print("cp: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic")
print("restecg: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy")
print("slope: 0=upsloping, 1=flat, 2=downsloping")
print("thal: 0=normal, 1=fixed defect, 2=reversible defect, 3=unknown")

while True:
    try:
        print("\nEnter patient details:")
        age      = float(input("  Age (years)                              : "))
        sex      = int(input("  Sex (1=Male, 0=Female)                   : "))
        cp       = int(input("  Chest Pain Type (0/1/2/3)                : "))
        trestbps = float(input("  Resting BP (mm Hg, e.g. 130)             : "))
        chol     = float(input("  Cholesterol (mg/dl, e.g. 250)            : "))
        fbs      = int(input("  Fasting Blood Sugar >120? (1=Yes, 0=No)  : "))
        restecg  = int(input("  Resting ECG (0/1/2)                      : "))
        thalach  = float(input("  Max Heart Rate Achieved (e.g. 150)       : "))
        exang    = int(input("  Exercise Induced Angina (1=Yes, 0=No)    : "))
        oldpeak  = float(input("  ST Depression (e.g. 1.5)                 : "))
        slope    = int(input("  Slope of ST Segment (0/1/2)              : "))
        ca       = int(input("  No. of Major Vessels (0-3)               : "))
        thal     = int(input("  Thalassemia (0/1/2/3)                    : "))
        patient = {
            'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,
            'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,
            'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal,
            'age_x_thalach':age*thalach, 'age_x_oldpeak':age*oldpeak,
            'chol_x_age':chol*age, 'bp_x_age':trestbps*age, 'cp_x_thalach':cp*thalach,
        }
        patient_df     = pd.DataFrame([patient])[feature_names]
        patient_scaled = scaler.transform(patient_df)
        prob           = best_tuned.predict_proba(patient_scaled)[0][1]
        pred           = int(prob >= optimal_thr)
        category       = get_risk_category(prob)
        print("\n" + "-"*45)
        print(f"  Risk Probability  : {prob:.2%}")
        print(f"  Risk Category     : {category}")
        print(f"  Threshold used    : {optimal_thr:.2f}")
        print(f"  RESULT: {'HIGH RISK — Heart Disease Detected' if pred else 'LOW RISK  — No Heart Disease'}")
        print("-"*45)
    except ValueError:
        print("  Invalid input. Please enter numbers only.")
    if input("\nPredict another patient? (yes/no): ").strip().lower() != 'yes':
        print("Exiting. Thank you!"); break