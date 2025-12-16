import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to training data CSV')
    parser.add_argument('models_dir', help='Directory to save models')
    args = parser.parse_args()

    # Загрузка параметров из params.yaml
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    rf_params = params.get('random_forest', {})
    lr_params = params.get('logistic_regression', {})
    nb_params = params.get('naive_bayes', {})

    # Загрузка данных
    train_df = pd.read_csv(args.input_path)
    y = train_df['label']
    X = train_df.drop('label', axis=1)

    # Обучение моделей
    rf = RandomForestClassifier(**rf_params, random_state=42)
    lr = LogisticRegression(**lr_params, max_iter=1000, random_state=42)
    nb = GaussianNB(**nb_params)

    rf.fit(X, y)
    lr.fit(X, y)
    nb.fit(X, y)

    # Сохранение моделей
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/rf_model.pkl')
    joblib.dump(lr, 'models/lr_model.pkl')
    joblib.dump(nb, 'models/nb_model.pkl')
    print('Models trained and saved to models/')

if __name__ == '__main__':
    main()