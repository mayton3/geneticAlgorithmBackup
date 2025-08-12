import numpy as np
import os
import sys
import datetime
import pytz
from typing import Optional, Tuple
from google.colab import drive

# Монтируем Google Диск (для Colab)
drive.mount('/content/drive')

# Директория для автосохранения
backupFolder = '/content/drive/MyDrive/geneticAlgorithmBackup'
os.makedirs(backupFolder, exist_ok=True)

def fitness(population: np.ndarray) -> np.ndarray:
    mask_negative = (population < 0).any(axis=1)
    cubes = population.astype(np.float64) ** 3
    fitness_values = np.abs(np.sum(cubes, axis=1))
    fitness_values[mask_negative] = np.inf
    return fitness_values

def generate_population(size: int, x_range: Tuple[int, int]) -> np.ndarray:
    return np.random.randint(x_range[0], x_range[1] + 1, size=(size, 3))

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    point = np.random.randint(1, 3)
    child = np.empty(3, dtype=int)
    child[:point] = parent1[:point]
    child[point:] = parent2[point:]
    return child

def mutate(individual: np.ndarray, x_range: Tuple[int, int], mutation_rate: float = 0.1) -> np.ndarray:
    for i in range(3):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.randint(x_range[0], x_range[1] + 1)
    return individual

def save_population_to_csv(population: np.ndarray, filename: str = "population_backup.csv") -> None:
    path = os.path.join(backupFolder, filename)
    np.savetxt(path, population, delimiter=",", fmt='%d')
    print(f"\nТекущее состояние популяции сохранено в файл '{path}'.")

def load_population_from_csv(filename: str = "population_backup.csv") -> Optional[np.ndarray]:
    path = os.path.join(backupFolder, filename)
    if os.path.exists(path):
        population = np.loadtxt(path, delimiter=",", dtype=int)
        if population.ndim == 1:
            population = population.reshape(1, -1)
        print(f"Загружена популяция из файла '{path}', размер: {population.shape}.")
        return population
    else:
        return None

def save_generation_to_csv(generation: int, filename: str = "generation_backup.csv") -> None:
    path = os.path.join(backupFolder, filename)
    np.savetxt(path, np.array([generation]), fmt='%d')
    print(f"Номер поколения {generation} сохранён в файл '{path}'.")

def load_generation_from_csv(filename: str = "generation_backup.csv") -> int:
    path = os.path.join(backupFolder, filename)
    if os.path.exists(path):
        gen_arr = np.loadtxt(path, dtype=int)
        if gen_arr.ndim == 0:
            generation = int(gen_arr)
        else:
            generation = int(gen_arr[0])
        print(f"Загружен номер поколения: {generation} из файла '{path}'.")
        return generation
    else:
        return 0

def genetic_algorithm(
    fitness,
    generate_population,
    crossover,
    mutate,
    population_size: int,
    x_range: Tuple[int, int],
    generations: int,
    mutation_rate: float = 0.1
) -> np.ndarray:
    population = load_population_from_csv()
    start_gen = load_generation_from_csv()

    if population is None:
        print("Файл с сохранённой популяцией не найден. Создаём новую популяцию.")
        population = generate_population(population_size, x_range)
        start_gen = 0
    else:
        print(f"Продолжаем работу с загруженной популяцией размером {population.shape[0]} и поколением {start_gen}")
        if population.shape[0] < population_size:
            needed = population_size - population.shape[0]
            new_individuals = generate_population(needed, x_range)
            population = np.vstack([population, new_individuals])

    best_solution = None
    best_fitness_value = np.inf

    try:
        for gen in range(start_gen, generations):
            fitness_values = fitness(population)

            # Отбор "интересных" хромосом - отличных от [0,0,0]
            mask_non_trivial = ~np.all(population == 0, axis=1)
            filtered_indices = np.where(mask_non_trivial)[0]

            if len(filtered_indices) > 0:
                filtered_fitness = fitness_values[filtered_indices]
                min_idx_filtered = filtered_indices[np.argmin(filtered_fitness)]
                min_fitness_filtered = fitness_values[min_idx_filtered]

                if min_fitness_filtered < best_fitness_value:
                    best_solution = population[min_idx_filtered].copy()
                    best_fitness_value = min_fitness_filtered

            best_idx = np.argmin(fitness_values)
            best_fitness = fitness_values[best_idx]

            timezone = pytz.timezone('Europe/Kiev')  # при необходимости замените
            current_time = datetime.datetime.now(pytz.utc).astimezone(timezone).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Поколение {gen + 1}, лучшая оценка (всех): {best_fitness}, "
                  f"лучшая оценка (интересных): {best_fitness_value if best_solution is not None else 'N/A'}")

            if best_fitness_value == 0:
                print("Найдено точное не тривиальное решение!")
                return best_solution

            sorted_indices = np.argsort(fitness_values)
            population = population[sorted_indices]

            next_generation = list(population[:2].copy())
            while len(next_generation) < population_size:
                parent_pool_size = min(10, len(population))
                parents_idx = np.random.choice(parent_pool_size, size=2, replace=False)
                parent1 = population[parents_idx[0]]
                parent2 = population[parents_idx[1]]
                child = crossover(parent1, parent2)
                child = mutate(child, x_range, mutation_rate)
                next_generation.append(child)

            population = np.array(next_generation)

            # Автосохранение каждые 3 поколения
            if (gen + 1) % 3 == 0:
                save_population_to_csv(population)
                save_generation_to_csv(gen + 1)

    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C). Сохраняем состояние и завершаем работу...")
        save_population_to_csv(population)
        save_generation_to_csv(gen)
        sys.exit(0)

    if best_solution is not None:
        return best_solution
    else:
        # Если не найдено ни одного "интересного" решения, возвращаем просто лучшее из всех
        fitness_values = fitness(population)
        best_idx = np.argmin(fitness_values)
        return population[best_idx]

if __name__ == "__main__":
    # Настройка параметров
    population_size = 50000000
    x_range = (-5000000, 5000000)
    generations = 1000

    solution = genetic_algorithm(
        fitness,
        generate_population,
        crossover,
        mutate,
        population_size,
        x_range,
        generations
    )

    print(f"\nНайденное решение (интересное): a={solution[0]}, b={solution[1]}, c={solution[2]}")
    print(f"Значение функции оценки: {np.abs(solution[0]**3 + solution[1]**3 + solution[2]**3)}")
