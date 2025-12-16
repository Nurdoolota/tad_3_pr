import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from datetime import datetime

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_metrics_path', help='Base path for metrics JSON (e.g., metrics/eval.json or metrics/eval_aug.json)')
    parser.add_argument('models_dir', help='Directory with trained models')
    args = parser.parse_args()

    # Загрузка тестовых данных
    test_df = pd.read_csv('data/raw/fashion-mnist_test.csv')
    y_test = test_df['label']
    X_test = test_df.drop('label', axis=1)

    # Загрузка моделей
    rf_path = os.path.join(args.models_dir, 'rf_model.pkl')
    lr_path = os.path.join(args.models_dir, 'lr_model.pkl')
    nb_path = os.path.join(args.models_dir, 'nb_model.pkl')
    
    rf = joblib.load(rf_path)
    lr = joblib.load(lr_path)
    nb = joblib.load(nb_path)

    # Предсказания и метрики
    metrics = {
        'random_forest': compute_metrics(y_test, rf.predict(X_test)),
        'logistic_regression': compute_metrics(y_test, lr.predict(X_test)),
        'naive_bayes': compute_metrics(y_test, nb.predict(X_test))
    }

    base_dir = os.path.dirname(args.base_metrics_path)
    new_metrics_path = os.path.join(base_dir, f"{'eval_aug.json'}")

    # Сохранение в новый файл
    os.makedirs(base_dir, exist_ok=True)
    with open(new_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f'Metrics saved to new file: {new_metrics_path}')
    print(f'(Original base path preserved: {args.base_metrics_path})')

if __name__ == '__main__':
    main()