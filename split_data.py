import os
import shutil
import random
from tqdm import tqdm  # Keeps track of progress in the terminal

# 1. Настройки исходных и новых папок
source_images_dir = "./data/train/images"  # Ваша текущая папка с картинками
source_labels_dir = "./data/train/labels"  # Ваша текущая папка с разметкой txt

output_base_dir = "./data"   # Главная папка для нового датасета

# Пути для новых папок
train_images_dir = os.path.join(output_base_dir, "train/images")
train_labels_dir = os.path.join(output_base_dir, "train/labels")
val_images_dir = os.path.join(output_base_dir, "val/images")
val_labels_dir = os.path.join(output_base_dir, "val/labels")

# Создаем структуру папок, если она еще не существует
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 2. Поиск пар файлов
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
all_images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(valid_extensions)]

paired_dataset = []
missing_labels_count = 0

for img_name in all_images:
    base_name, img_ext = os.path.splitext(img_name)
    label_name = f"{base_name}.txt"
    label_path = os.path.join(source_labels_dir, label_name)
    
    # Проверяем, существует ли файл аннотации для этой картинки
    if os.path.exists(label_path):
        paired_dataset.append((img_name, label_name))
    else:
        missing_labels_count += 1

print(f"Found {len(paired_dataset)} valid image-label pairs.")
if missing_labels_count > 0:
    print(f"Warning: {missing_labels_count} images skipped because their .txt label file was missing.")

# 3. Перемешивание и расчет пропорций (9:1)
random.seed(42)  # Фиксируем seed, чтобы результат разделения был воспроизводимым
random.shuffle(paired_dataset)

split_index = int(len(paired_dataset) * 0.9)  # 90% уходит в train
train_pairs = paired_dataset[:split_index]
val_pairs = paired_dataset[split_index:]

print(f"Splitting: {len(train_pairs)} pairs for Train | {len(val_pairs)} pairs for Val.")

# Функция для копирования файлов
def copy_files(pairs, dest_img_dir, dest_lbl_dir, desc):
    for img_name, lbl_name in tqdm(pairs, desc=desc):
        # Исходные пути
        src_img = os.path.join(source_images_dir, img_name)
        src_lbl = os.path.join(source_labels_dir, lbl_name)
        
        # Новые пути назначения
        dst_img = os.path.join(dest_img_dir, img_name)
        dst_lbl = os.path.join(dest_lbl_dir, lbl_name)
        
        # Копирование
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

# 4. Выполнение копирования
print("\nCopying Train data...")
copy_files(train_pairs, train_images_dir, train_labels_dir, "Train progress")

print("\nCopying Val data...")
copy_files(val_pairs, val_images_dir, val_labels_dir, "Val progress")

print(f"\nSuccess! Your dataset is split and saved to: {os.path.abspath(output_base_dir)}")
