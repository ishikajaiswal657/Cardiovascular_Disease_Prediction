# Heart Disease Prediction Using Machine Learning

A complete machine learning pipeline for binary prediction of heart disease
using the Cleveland Heart Disease Dataset.

## Dataset
- **Name:** Cleveland Heart Disease Dataset
- **Records:** 303 patient records
- **Features:** 14 clinical features
- **Source:** UCI Machine Learning Repository / Kaggle

## Final Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 91.67% | 0.9520 |
| Neural Network (Keras) | 88.33% | 0.9632 |
| Ensemble VotingClassifier | 86.67% | 0.9609 |
| After Threshold Tuning | 91.67% | 0.9632 |

## Methods Applied
1. Z-Score Normalization
2. SMOTE Data Augmentation
3. 10-Fold Stratified Cross-Validation
4. F1 Score Optimization (Threshold Tuning)
5. Hyperparameter Tuning (RandomizedSearchCV + Optuna 100 trials)
6. Keras Neural Network (Dropout + BatchNormalization + EarlyStopping)
7. Ensemble VotingClassifier (Soft Voting)

## How to Run
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost tensorflow optuna matplotlib seaborn joblib
python main.py
```

## Project Structure
| File | Description |
|---|---|
| `main.py` | Complete ML pipeline |
| `heart_cleveland_upload.csv` | Cleveland dataset |
| `heart_model.pkl` | Saved trained model |
| `heart_scaler.pkl` | Saved Z-Score scaler |
| `heart_threshold.pkl` | Optimal classification threshold |
| `heart_features.pkl` | Saved feature names |

## Output Charts
- `roc_curves.png` — ROC curves for all models
- `confusion_matrix.png` — Confusion matrix of final model
- `feature_importance.png` — Feature importance (Random Forest)
- `neural_network_training.png` — NN loss and accuracy curves
- `threshold_tuning.png` — F1 score vs threshold
- `optuna_tuning.png` — Optuna 100 trial AUC history
- `risk_categories.png` — Patient risk category distribution
- `eda_cleveland.png` — EDA plots

