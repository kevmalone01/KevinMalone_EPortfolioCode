"""
A*-Algorithmus (A Star) – Pfadsuche in einem Grid
------------------------------------------------

Dieses Programm zeigt, wie der A*-Algorithmus einen kürzesten Pfad in einem 2D-Gitter (Grid) findet.
Es ist ein reines Konsolenprogramm und eignet sich ideal, um die Logik von A* nachzuvollziehen.

Was macht das Programm?
-----------------------
- Es sucht den optimalen Weg von einem Startpunkt zu einem Zielpunkt in einem Grid mit Hindernissen.
- Das Grid ist als 2D-Liste aufgebaut (0 = frei, 1 = Hindernis).
- Die Ausgabe ist eine Liste der Koordinaten, die den gefundenen Pfad darstellen.

Wie funktioniert der Code?
--------------------------
1. **Node-Klasse:**
   - Jeder Knoten (Node) speichert seine Position, die Kosten vom Start (g), die geschätzten Kosten zum Ziel (h) und die Gesamtkosten (f = g + h).
   - Jeder Knoten merkt sich seinen Vorgänger (parent), um den Pfad später rekonstruieren zu können.

2. **Heuristik:**
   - Die Funktion `heuristic` berechnet die Manhattan-Distanz zwischen zwei Punkten (Start und Ziel).

3. **Nachbarn finden:**
   - Die Funktion `get_neighbors` gibt alle gültigen Nachbarfelder (oben, rechts, unten, links) zurück, die keine Hindernisse sind.

4. **A*-Algorithmus (`astar`):**
   - Start- und Zielknoten werden angelegt.
   - Die offene Liste (open_set) enthält alle zu prüfenden Knoten (als Min-Heap nach f-Kosten sortiert).
   - Die geschlossene Liste (closed_set) enthält alle bereits geprüften Knoten.
   - In jeder Runde wird der Knoten mit den niedrigsten f-Kosten aus der offenen Liste genommen.
   - Für jeden Nachbarn wird geprüft, ob ein besserer Weg gefunden wurde. Falls ja, werden die Kosten und der Vorgänger aktualisiert.
   - Wenn das Ziel erreicht ist, wird der Pfad rekonstruiert und zurückgegeben.
   - Falls kein Pfad gefunden wird, gibt die Funktion eine leere Liste zurück.
"""

import heapq
from typing import List, Tuple, Set, Dict

class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0):
        self.position = position
        self.g_cost = g_cost  # Kosten vom Startknoten
        self.h_cost = h_cost  # Geschätzte Kosten zum Zielknoten
        self.f_cost = g_cost + h_cost  # Gesamtkosten
        self.parent = None

    def __lt__(self, other):
        return self.f_cost < other.f_cost

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Berechnet die Distanz zwischen zwei Punkten."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[int, int]]:
    """Gibt die gültigen Nachbarn einer Position zurück."""
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Rechts, Unten, Links, Oben
    
    for dx, dy in directions:
        new_x, new_y = pos[0] + dx, pos[1] + dy
        
        # Prüfe, ob die neue Position innerhalb des Grids liegt und begehbar ist
        if (0 <= new_x < len(grid) and 
            0 <= new_y < len(grid[0]) and 
            grid[new_x][new_y] != 1):  # 1 repräsentiert Hindernisse
            neighbors.append((new_x, new_y))
    
    return neighbors

def astar(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Implementierung des A* Algorithmus.
    
    Args:
        grid: 2D-Liste, wobei 0 begehbare Felder und 1 Hindernisse repräsentiert
        start: Startposition (x, y)
        end: Zielposition (x, y)
    
    Returns:
        Liste von Positionen, die den Pfad vom Start zum Ziel darstellen
    """
    # Initialisiere offene und geschlossene Listen
    open_set: List[Node] = []
    closed_set: Set[Tuple[int, int]] = set()
    
    # Erstelle Start- und Endknoten
    start_node = Node(start, 0, heuristic(start, end))
    end_node = Node(end)
    
    # Füge Startknoten zur offenen Liste hinzu
    heapq.heappush(open_set, start_node)
    
    # Dictionary für schnellen Zugriff auf Knoten
    node_dict: Dict[Tuple[int, int], Node] = {start: start_node}
    
    while open_set:
        # Hole Knoten mit niedrigsten f-Kosten
        current = heapq.heappop(open_set)
        
        # Wenn wir das Ziel erreicht haben
        if current.position == end:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Kehre den Pfad um
        
        # Füge aktuellen Knoten zur geschlossenen Liste hinzu
        closed_set.add(current.position)
        
        # Prüfe alle Nachbarn
        for neighbor_pos in get_neighbors(current.position, grid):
            if neighbor_pos in closed_set:
                continue
            
            # Berechne neue g-Kosten
            new_g_cost = current.g_cost + 1
            
            # Wenn der Nachbar noch nicht in der offenen Liste ist oder
            # wenn wir einen besseren Pfad gefunden haben
            if neighbor_pos not in node_dict or new_g_cost < node_dict[neighbor_pos].g_cost:
                neighbor = Node(
                    neighbor_pos,
                    new_g_cost,
                    heuristic(neighbor_pos, end)
                )
                neighbor.parent = current
                node_dict[neighbor_pos] = neighbor
                heapq.heappush(open_set, neighbor)
    
    return []  # Kein Pfad gefunden

# Beispiel zur Verwendung
if __name__ == "__main__":
    # Beispiel-Grid (0 = begehbar, 1 = Hindernis)
    grid = [
        [0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0]
    ]
    
    start_pos = (0, 0)
    end_pos = (4, 4)
    
    path = astar(grid, start_pos, end_pos)
    
    if path:
        print("Gefundener Pfad:")
        for pos in path:
            print(f"-> {pos}")
    else:
        print("Kein Pfad gefunden!") 