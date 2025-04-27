G = Graph()
edges = [(0,1,1), (1,2,2), (2,3,1), (0,3,4), (1,4,3), (4,3,2), (2,5,2), (5,6,3), (6,7,1), (5,7,2), (3,7,4), (4,8,5), (8,9,2), (9,7,3)]
for a, b, cost in edges:
    G.add(a, b, cost)

heuristics = {0:5, 1:4, 2:3, 3:2, 4:4, 5:3, 6:2, 7:1, 8:3, 9:0}
for node, value in heuristics.items():
    G.h(node, value)

G.visualize()
G.a_star(0, 9)

A: undirected:
import networkx as nx
import matplotlib.pyplot as plt
import heapq

class gv:
    def __init__(self):
        self.visual = []
        self.adj_list = {}
        self.heuristic = {}

    def addEdge(self, a, b, cost):
        self.visual.append([a, b])
        if a not in self.adj_list:
            self.adj_list[a] = []
        if b not in self.adj_list:
            self.adj_list[b] = []
        self.adj_list[a].append((b, cost))
        self.adj_list[b].append((a, cost))

    def setHeuristic(self, n, value):
        self.heuristic[n] = value

    def visualize(self):
        grp = nx.Graph()
        grp.add_edges_from(self.visual)
        nx.draw_networkx(grp)
        plt.show()

    def a_star(self, start, goal):
        pq = []
        heapq.heappush(pq, (self.heuristic[start], 0, start))
        visited = set()
        parent = {start: None}
        cost_so_far = {start: 0}

        while pq:
            est_total, cost, n = heapq.heappop(pq)
            if n in visited:
                continue

            visited.add(n)

            if n == goal:
                break

            for neighbor, weight in self.adj_list.get(n, []):
                new_cost = cost + weight
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic.get(neighbor, float('inf'))
                    heapq.heappush(pq, (priority, new_cost, neighbor))
                    parent[neighbor] = n

        self.visualize_tree(parent)
        print("A* Path:", self.construct_path(parent, start, goal))

    def visualize_tree(self, parent):
        tree_edges = [(parent[n], n) for n in parent if parent[n] is not None]
        tree = nx.Graph()
        tree.add_edges_from(tree_edges)
        nx.draw_networkx(tree)
        plt.show()

    def construct_path(self, parent, start, goal):
        path = []
        while goal is not None:
            path.append(goal)
            goal = parent[goal]
        return path[::-1]

G = gv()

n_edges = int(input("Enter number of edges: "))
for _ in range(n_edges):
    a = int(input("Enter node A: "))
    b = int(input("Enter node B: "))
    cost = int(input("Enter weight: "))
    G.addEdge(a, b, cost)

n_nodes = int(input("Enter number of nodes for heuristic: "))
for _ in range(n_nodes):
    node = int(input("Enter node number: "))
    value = int(input("Enter heuristic value: "))
    G.setHeuristic(node, value)

G.visualize()
start = int(input("Enter start node: "))
goal = int(input("Enter goal node: "))
G.a_star(start, goal)

CSV:
df = pd.read_csv('graph_data.csv')

G = gv()

# Add Edges
for idx, row in df.iterrows():
    G.addEdge(row['source'], row['target'], row['weight'])

# Add Heuristics
for idx, row in df.iterrows():
    if not pd.isna(row['heuristic']):
        G.setHeuristic(int(row['source']), int(row['heuristic']))

G.visualize()
G.a_star(0, 9)

B: directed :
import networkx as nx
import matplotlib.pyplot as plt
import heapq

class gv:
    def __init__(self):
        self.visual = []
        self.adj_list = {}
        self.heuristic = {}

    def addEdge(self, a, b, cost):
        self.visual.append([a, b])
        if a not in self.adj_list:
            self.adj_list[a] = []
        self.adj_list[a].append((b, cost))

    def setHeuristic(self, n, value):
        self.heuristic[n] = value

    def visualize(self):
        grp = nx.DiGraph()
        grp.add_edges_from(self.visual)
        nx.draw_networkx(grp)
        plt.show()

    def a_star(self, start, goal):
        pq = []
        heapq.heappush(pq, (self.heuristic[start], 0, start))
        visited = set()
        parent = {start: None}
        cost_so_far = {start: 0}

        while pq:
            est_total, cost, n = heapq.heappop(pq)
            if n in visited:
                continue

            visited.add(n)

            if n == goal:
                break

            for neighbor, weight in self.adj_list.get(n, []):
                new_cost = cost + weight
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic.get(neighbor, float('inf'))
                    heapq.heappush(pq, (priority, new_cost, neighbor))
                    parent[neighbor] = n

        self.visualize_tree(parent)
        print("A* Path:", self.construct_path(parent, start, goal))

    def visualize_tree(self, parent):
        tree_edges = [(parent[n], n) for n in parent if parent[n] is not None]
        tree = nx.DiGraph()
        tree.add_edges_from(tree_edges)
        nx.draw_networkx(tree)
        plt.show()

    def construct_path(self, parent, start, goal):
        path = []
        while goal is not None:
            path.append(goal)
            goal = parent[goal]
        return path[::-1]

G = gv()

n_edges = int(input("Enter number of edges: "))
for _ in range(n_edges):
    a = int(input("Enter source node: "))
    b = int(input("Enter target node: "))
    cost = int(input("Enter weight: "))
    G.addEdge(a, b, cost)

n_nodes = int(input("Enter number of nodes for heuristic: "))
for _ in range(n_nodes):
    node = int(input("Enter node number: "))
    value = int(input("Enter heuristic value: "))
    G.setHeuristic(node, value)

G.visualize()
start = int(input("Enter start node: "))
goal = int(input("Enter goal node: "))
G.a_star(start, goal)

CSV:
df = pd.read_csv('graph_data.csv')

G = gv()

# Add Edges
for idx, row in df.iterrows():
    G.addEdge(row['source'], row['target'], row['weight'])

# Add Heuristics
for idx, row in df.iterrows():
    if not pd.isna(row['heuristic']):
        G.setHeuristic(int(row['source']), int(row['heuristic']))

G.visualize()
G.a_star(0, 9)
