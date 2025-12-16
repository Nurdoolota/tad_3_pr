import pandas as pd
import numpy as np
import argparse
import os
from scipy.ndimage import rotate, shift

def apply_transformations(image_vector):
    """
    Принимает вектор 784 пикселя (масштабированный в [0, 1]),
    возвращает список из 5 аугментированных векторов без NaN и в диапазоне [0, 1].
    """
    img = image_vector.reshape(28, 28)
    
    augmented = []
    
    # 1. Ротация +10°
    rot1 = rotate(img, angle=10, reshape=False, mode='constant', cval=0.0)
    rot1 = np.nan_to_num(rot1, nan=0.0)
    augmented.append(np.clip(rot1, 0.0, 1.0).flatten())
    
    # 2. Ротация -10°
    rot2 = rotate(img, angle=-10, reshape=False, mode='constant', cval=0.0)
    rot2 = np.nan_to_num(rot2, nan=0.0)
    augmented.append(np.clip(rot2, 0.0, 1.0).flatten())
    
    # 3. Сдвиг вверх-влево (-2, -2)
    shift1 = shift(img, shift=(-2, -2), mode='constant', cval=0.0)
    augmented.append(np.clip(shift1, 0.0, 1.0).flatten())
    
    # 4. Сдвиг вниз-вправо (2, 2)
    shift2 = shift(img, shift=(2, 2), mode='constant', cval=0.0)
    augmented.append(np.clip(shift2, 0.0, 1.0).flatten())
    
    # 5. Горизонтальное отражение + шум
    flipped = np.fliplr(img)
    noise = np.random.normal(0, 0.05, flipped.shape)
    noisy = np.clip(flipped + noise, 0.0, 1.0)
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

    # Нормализация в [0, 1] (Fashion MNIST обычно в [0, 255], делим на 255)
    pixels /= 255.0

    augmented_images = []
    augmented_labels = []

    for i in range(len(pixels)):
        transforms = apply_transformations(pixels[i])
        augmented_images.extend(transforms)
        augmented_labels.extend([labels[i]] * len(transforms))

    # Создание DataFrame
    columns = sampled_df.columns[1:] 
    aug_df = pd.DataFrame(augmented_images, columns=columns)
    aug_df.insert(0, 'label', augmented_labels)

    # Обратная денормализация в [0, 255] для совместимости с исходным форматом
    for col in columns:
        aug_df[col] *= 255.0
        aug_df[col] = aug_df[col].round().astype(int)

    # Сохранение
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    aug_df.to_csv(args.output_path, index=False)
    print(f'Augmented dataset (only copies, {len(aug_df)} rows) saved to {args.output_path}')

if __name__ == '__main__':
    main()