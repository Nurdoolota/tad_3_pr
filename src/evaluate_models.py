import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_path', help='Path to save metrics JSON')
    args = parser.parse_args()

    # Загрузка тестовых данных
    test_df = pd.read_csv('data/raw/fashion-mnist_test.csv')
    y_test = test_df['label']
    X_test = test_df.drop('label', axis=1)

    # Загрузка моделей
    rf = joblib.load('models/rf_model.pkl')
    lr = joblib.load('models/lr_model.pkl')
    nb = joblib.load('models/nb_model.pkl')

    # Предсказания и метрики
    metrics = {
        'random_forest': compute_metrics(y_test, rf.predict(X_test)),
        'logistic_regression': compute_metrics(y_test, lr.predict(X_test)),
        'naive_bayes': compute_metrics(y_test, nb.predict(X_test))
    }

    # Запись в JSON (создание файла, если отсутствует; иначе перезапись)
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'Metrics saved to {args.metrics_path}')

if __name__ == '__main__':
    main()