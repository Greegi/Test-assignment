import random
import numpy as np
import matplotlib.pyplot as plt

class CityGrid:
    def __init__(self, N, M, obstructed_prob=0.3):
        self.N = N
        self.M = M
        self.grid = np.zeros((N, M), dtype=int)

        # Инициализируем блоки с преградами
        for i in range(N):
            for j in range(M):
                if random.random() < obstructed_prob:
                    self.grid[i][j] = 1  # Помечаем как преграду

    def place_tower(self, x, y, range_R):
        # Размещаем башню и визуализируем ее зону покрытия
        for i in range(max(0, x - range_R), min(self.N, x + range_R + 1)):
            for j in range(max(0, y - range_R), min(self.M, y + range_R + 1)):
                if (x - i) ** 2 + (y - j) ** 2 <= range_R ** 2 and self.grid[i][j] != 1:
                    self.grid[i][j] = 2  # Помечаем как покрытие башни

    def optimize_tower_placement(self, range_R):
        # Реализуем алгоритм размещения башен, чтобы покрыть непрегражденные блоки
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j] != 1:
                    self.place_tower(i, j, range_R)

    def find_reliable_path(self, tower1, tower2):
        return None
        # Реализуем алгоритм поиска надежного пути между двумя башнями
        # Можно использовать графовые алгоритмы, такие как Дейкстра или A*, чтобы найти путь

    def visualize_city(self):
        # Визуализируем CityGrid с использованием Matplotlib
        plt.imshow(self.grid, cmap='viridis', interpolation='nearest')
        plt.show()

    def visualize_grid(self):
        plt.imshow(self.grid, cmap='Greys', interpolation='none')
        plt.title("City Grid")
        plt.show()

    def visualize_towers(self):
        # Visualize tower placement and coverage
        pass

    def visualize_reliable_path(self):
        # Visualize the reliable path between towers
        pass

# Пример использования
city = CityGrid(10, 10)
city.optimize_tower_placement(3)
city.visualize_city()