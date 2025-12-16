import pandas as pd
import argparse
import yaml
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to save sampled data')
    args = parser.parse_args()

    # Загрузка params.yaml для получения sample_size
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    sample_size = params.get('sample_size', 5000)

    # Загрузка исходных данных
    train_df = pd.read_csv('data/raw/fashion-mnist_train.csv')
    
    # Выборка случайных sample_size строк
    sampled_df = train_df.sample(n=sample_size, random_state=42)
    
    # Сохранение в CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    sampled_df.to_csv(args.output_path, index=False)
    print(f'Sampled {sample_size} rows saved to {args.output_path}')

if __name__ == '__main__':
    main()