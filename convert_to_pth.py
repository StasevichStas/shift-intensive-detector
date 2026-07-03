import torch

# Путь к вашему чекпоинту
ckpt_path = "/root/shift-intensive-detector/logs/train/baseline/runs/2026-07-03_14-40-05/checkpoints/last.ckpt"  # или полный путь /root/shift-intensive-detector/logs/.../last.ckpt
output_path = "model.pth"

# 1. Загружаем чекпоинт (map_location='cpu' чтобы не тратить VRAM)
checkpoint = torch.load(ckpt_path, map_location="cpu")

# 2. Извлекаем state_dict
state_dict = checkpoint['state_dict']

# 3. Проверяем ключи на наличие префикса 'model.'
# Это частая проблема в Lightning-модулях
sample_key = list(state_dict.keys())[0]
print(f"Первый ключ: {sample_key}")

if sample_key.startswith("model."):
    print("Обнаружен префикс 'model.', удаляю его...")
    clean_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
else:
    clean_state_dict = state_dict

# 4. Сохраняем чистые веса
torch.save(clean_state_dict, output_path)

print(f"✅ Конвертация завершена!")
print(f"   Файл: {output_path}")
print(f"   Количество тензоров: {len(clean_state_dict)}")
print(f"   Размер файла: {__import__('os').path.getsize(output_path) / 1024 / 1024:.2f} MB")
