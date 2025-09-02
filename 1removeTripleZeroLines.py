import os
import csv
from datetime import datetime

input_file = "population_backup.csv"
output_file = "population_backup_cleaned.csv"
report_file = "report1.csv"

# Удаляем строки "0,0,0" и сохраняем
with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        if line.strip() != "0,0,0":
            fout.write(line)

# Считаем количество строк в новом файле
with open(output_file, "r") as f:
    line_count = sum(1 for _ in f)

# Получаем размер файла в байтах и мегабайтах
file_size_bytes = os.path.getsize(output_file)
file_size_mb = file_size_bytes / (1024 * 1024)

# Текущая дата и время в нужном формате
current_time = datetime.now().strftime("%d.%m.%Y %H:%M")

# Проверяем существует ли файл отчета, если нет — записываем заголовок
file_exists = os.path.exists(report_file)

with open(report_file, mode="a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["Количество строк", "Дата и время", "Размер файла"])
    writer.writerow([line_count, current_time, f"{file_size_mb:.0f} МБ ({file_size_bytes} байт)"])

print("Готово.")
print(f"{line_count} | {current_time} | {file_size_mb:.0f} МБ ({file_size_bytes} байт)")
