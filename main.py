import random
import numpy as np
import matplotlib.pyplot as plt
import heapq


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
        # Проверяем, являются ли башни валидными
        if self.grid[tower1[0]][tower1[1]] != 2 or self.grid[tower2[0]][tower2[1]] != 2:
            return None  # Одна из башен не в пределах покрытия

        # Граф в виде словаря, где ключи - вершины (башни), а значения - соседи (ближайшие башни)
        graph = {}
        for i in range(self.N):
            for j in range(self.M):
                if self.grid[i][j] == 2:
                    neighbors = []
                    for x in range(max(0, i - 1), min(self.N, i + 2)):
                        for y in range(max(0, j - 1), min(self.M, j + 2)):
                            if self.grid[x][y] == 2 and (x != i or y != j):
                                # Рассчитываем вес ребра на основе надежности
                                reliability = self.calculate_reliability((i, j), (x, y))
                                neighbors.append(((x, y), reliability))
                    graph[(i, j)] = neighbors

        # Используем алгоритм Дейкстры для поиска пути с учетом надежности
        dist = {}  # Минимальное расстояние от начальной вершины (башни) до остальных
        prev = {}  # Предыдущая вершина на пути к текущей вершине
        for vertex in graph:
            dist[vertex] = float('inf')
        dist[tower1] = 0
        queue = [(0, tower1)]

        while queue:
            current_dist, current_vertex = heapq.heappop(queue)
            if current_dist > dist[current_vertex]:
                continue

            for neighbor, reliability in graph[current_vertex]:
                weight = 1 / reliability  # Вес ребра обратно пропорционален надежности
                if dist[current_vertex] + weight < dist[neighbor]:
                    dist[neighbor] = dist[current_vertex] + weight
                    prev[neighbor] = current_vertex
                    heapq.heappush(queue, (dist[neighbor], neighbor))

        # Восстанавливаем путь
        path = []
        current = tower2
        while current in prev:
            path.insert(0, current)
            current = prev[current]

        if path:
            return path
        else:
            return None  # Нет пути между башнями

    def calculate_reliability(self, tower1, tower2):
        # Расчет надежности сигнала между башнями
        distance = self.calculate_distance(tower1, tower2)

        # Пример расчета надежности на основе расстояния (чем ближе, тем надежнее)
        reliability = 1 / (1 + distance)

        return reliability

    def calculate_distance(self, tower1, tower2):
        # Расчет расстояния между башнями (просто Евклидово расстояние)
        x1, y1 = tower1
        x2, y2 = tower2
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance

    def visualize_reliable_path(self, path):
        # Визуализация надежного пути между башнями
        if not path:
            return

        reliable_path = np.copy(self.grid)
        for x, y in path:
            reliable_path[x, y] = 3  # Помечаем путь

        plt.imshow(reliable_path, cmap='viridis', interpolation='nearest')
        plt.show()

    def visualize_city(self):
        # Визуализируем CityGrid с использованием Matplotlib
        plt.imshow(self.grid, cmap='viridis', interpolation='nearest')
        plt.show()

    def visualize_grid(self):
        plt.imshow(self.grid, cmap='viridis', interpolation='nearest')
        plt.show()

    def visualize(self, aspect='city', path=None):
        if aspect == 'city':
            plt.imshow(self.grid, cmap='viridis', interpolation='nearest')
            plt.show()
        elif aspect == 'reliable_path' and path:
            reliable_path = np.copy(self.grid)
            for x, y in path:
                reliable_path[x, y] = 3
            plt.imshow(reliable_path, cmap='viridis', interpolation='nearest')
            plt.show()


# Пример использования
city = CityGrid(10, 10)
city.optimize_tower_placement(20)
city.visualize('city')

path = city.find_reliable_path((3, 4), (6, 7))
city.visualize('reliable_path', path)

if path:
    print(f"Надежный путь: {path}")
else:
    print("Надежный путь не найден")
