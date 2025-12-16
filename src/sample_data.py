import pandas as pd
import argparse
import yaml
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to save combined dataset')
    args = parser.parse_args()

    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    # Загрузка оригинальных 5000 образцов
    original_df = pd.read_csv('data/sampled_train.csv')

    # Загрузка только аугментированных копий (результат стадии augment_data)
    augmented_df = pd.read_csv('data/augmented_only_train.csv')

    # Объединение оригинальных и аугментированных данных
    combined_df = pd.concat([original_df, augmented_df], ignore_index=True)

    # Перемешивание для равномерного распределения
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Сохранение объединённого датасета
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    combined_df.to_csv(args.output_path, index=False)
    print(f'Combined dataset ({len(original_df)} original + {len(augmented_df)} augmented = {len(combined_df)} rows) '
          f'saved to {args.output_path}')

if __name__ == '__main__':
    main()