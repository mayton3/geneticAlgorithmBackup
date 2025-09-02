import os
import csv
from datetime import datetime

def removeDoubleAndTripleZeroLines():
    input_file = "population_backup.csv"
    output_file = "population_backup_cleaned.csv"
    report_file = "report2.csv"

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            parts = line.strip().split(",")
            zero_count = parts.count("0")
            # Пропускаем строки с 2 и более нулями
            if zero_count < 2:
                fout.write(line)

    # Считаем количество строк и размер нового файла
    with open(output_file, "r") as f:
        line_count = sum(1 for _ in f)

    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)

    current_time = datetime.now().strftime("%d.%m.%Y %H:%M")

    file_exists = os.path.exists(report_file)

    with open(report_file, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Количество строк", "Дата и время", "Размер файла"])
        writer.writerow([line_count, current_time, f"{file_size_mb:.0f} МБ ({file_size_bytes} байт)"])

    print("Удаление строк с двумя и тремя нулями завершено.")
    print(f"{line_count} | {current_time} | {file_size_mb:.0f} МБ ({file_size_bytes} байт)")

if __name__ == "__main__":
    removeDoubleAndTripleZeroLines()
