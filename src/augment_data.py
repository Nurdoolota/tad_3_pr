import pandas as pd
import numpy as np
import argparse
import os
from scipy.ndimage import rotate, shift
from sklearn.preprocessing import MinMaxScaler

def apply_transformations(image_vector):
    """Принимает вектор 784 пикселя (нормализованный [0,1]), возвращает 5 аугментированных векторов без NaN"""
    img = image_vector.reshape(28, 28)
    
    augmented = []
    
    # # 1. Ротация +10°
    # rotated1 = rotate(img, angle=10, reshape=False, mode='constant', cval=0)
    # rotated1 = np.nan_to_num(rotated1, nan=0.0)  # Замена NaN на 0
    # augmented.append(rotated1.flatten())
    
    # # 2. Ротация -10°
    # rotated2 = rotate(img, angle=-10, reshape=False, mode='constant', cval=0)
    # rotated2 = np.nan_to_num(rotated2, nan=0.0)
    # augmented.append(rotated2.flatten())
    
    # # 3. Сдвиг вверх-влево
    # shifted1 = shift(img, shift=(-2, -2), mode='constant', cval=0)
    # augmented.append(shifted1.flatten())
    
    # # 4. Сдвиг вниз-вправо
    # shifted2 = shift(img, shift=(2, 2), mode='constant', cval=0)
    # augmented.append(shifted2.flatten())
    
    # 5. Горизонтальное отражение + шум
    flipped = np.fliplr(img)
    noise = np.random.normal(0, 0.05, flipped.shape)
    noisy = np.clip(flipped + noise, 0, 1)
    augmented.append(noisy.flatten())
    
    return augmented

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to save augmented data only')
    args = parser.parse_args()

    # Загрузка исходных 5000 образцов
    sampled_df = pd.read_csv('data/sampled_train.csv')
    labels = sampled_df['label'].values
    pixels = sampled_df.drop('label', axis=1).values.astype(np.float32)

    # Нормализация в [0, 1]
    scaler = MinMaxScaler()
    pixels = scaler.fit_transform(pixels)

    augmented_images = []
    augmented_labels = []

    for i in range(len(pixels)):
        transforms = apply_transformations(pixels[i])
        augmented_images.extend(transforms)
        augmented_labels.extend([labels[i]] * len(transforms))

    # DataFrame только с аугментированными данными
    columns = [f'pixel{i}' for i in range(784)]
    aug_df = pd.DataFrame(augmented_images, columns=columns)
    aug_df.insert(0, 'label', augmented_labels)

    # Сохранение
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    aug_df.to_csv(args.output_path, index=False)
    print(f'Augmented dataset (only copies, {len(aug_df)} rows) saved to {args.output_path}')

if __name__ == '__main__':
    main()