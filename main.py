import random
import numpy as np
import matplotlib.pyplot as plt
import heapq

class CityGrid:
    def __init__(self, N, M, obstructed_prob=0.3):
        self.N = N
        self.M = M
        self.grid = np.zeros((N, M), dtype=int)
        self.towers = []  # Список для хранения размещенных башен
        self.budget = 0

        # Инициализируем блоки с преградами
        for i in range(N):
            for j in range(M):
                if random.random() < obstructed_prob:
                    self.grid[i][j] = 1  # Помечаем как преграду

    def place_tower(self, x, y, range_R, cost):
        # Размещаем башню и визуализируем ее зону покрытия
        if cost <= self.budget:
            self.towers.append((x, y, range_R, cost))  # Добавляем башню в список
            self.budget -= cost
            for i in range(max(0, x - range_R), min(self.N, x + range_R + 1)):
                for j in range(max(0, y - range_R), min(self.M, y + range_R + 1)):
                    if (x - i) ** 2 + (y - j) ** 2 <= range_R ** 2 and self.grid[i][j] != 1:
                        self.grid[i][j] = 2  # Помечаем как покрытие башни
            print(f"Placed tower at ({x}, {y}) with range {range_R} and cost {cost}. Budget remaining: {self.budget}")

    def optimize_tower_placement(self, budget):
        self.budget = budget
        self.towers = []  # Сбросим уже размещенные башни
        radius_step = 2  # Радиус башни используется как шаг для равномерного размещения

        for i in range(0, self.N, radius_step):
            for j in range(0, self.M, radius_step):
                if self.grid[i][j] != 1:
                    # Рассчитываем стоимость башни в данной ячейке (просто случайное значение в данном примере)
                    cost = random.randint(1, 10)
                    # Рассчитываем радиус башни в данной ячейке (просто случайное значение в данном примере)
                    range_R = random.randint(1, 5)
                    self.place_tower(i, j, range_R, cost)

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
            print(f"Надежный путь от {tower1} к {tower2}: {path}")
            return path
        else:
            print(f"Надежный путь от {tower1} к {tower2} не найден.")
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

    def visualize(self, aspect='city', path=None):
        plt.figure()
        if aspect == 'city':
            plt.imshow(self.grid, cmap='viridis', interpolation='nearest', origin='lower')
        elif aspect == 'towers':
            plt.imshow(self.grid, cmap='viridis', interpolation='nearest', origin='lower')
            tower_locations = np.array([tower[:2] for tower in self.towers])
            plt.scatter(tower_locations[:, 1], tower_locations[:, 0], marker='s', c='r', label='Towers')
            for x, y, _, _ in self.towers:
                plt.text(y, x, f'Tower({y}, {x})', fontsize=8, color='r')
            plt.legend()
        elif aspect == 'reliable_path':
            reliable_path = np.copy(self.grid)
            for x, y in path:
                reliable_path[x, y] = 3
            plt.imshow(reliable_path, cmap='viridis', interpolation='nearest', origin='lower')
            for x, y in path:
                plt.text(y, x, f'Path({y}, {x})', fontsize=8, color='b')
        plt.show()

# Пример использования
city = CityGrid(10, 10)
budget = 50
city.optimize_tower_placement(budget)
city.visualize(aspect='city')

# Выберем две случайные башни для поиска надежного пути
tower1 = random.choice(city.towers)[:2]
tower2 = random.choice(city.towers)[:2]
city.visualize(aspect='towers')
path = city.find_reliable_path(tower1, tower2)

if path:
    city.visualize(aspect='reliable_path', path=path)