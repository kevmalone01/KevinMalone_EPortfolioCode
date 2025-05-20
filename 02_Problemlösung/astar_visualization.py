"""
A*-Algorithmus Visualisierung mit Pygame
----------------------------------------

Was macht das Programm?
Dieses Programm zeigt, wie der A*-Algorithmus einen kürzesten Pfad in einem Gitter (Grid) findet – und zwar interaktiv und visuell mit Pygame.

Wie funktioniert das Programm?
- Grid und Hindernisse:
  Du kannst mit der Maus ein Spielfeld (Grid) gestalten, indem du Hindernisse einzeichnest.
  - Linke Maustaste: Hindernisse setzen
  - Rechte Maustaste: Hindernisse entfernen
- Start- und Zielpunkt:
  - Shift + Linksklick: Startpunkt setzen
  - Strg + Linksklick: Zielpunkt setzen
- A*-Algorithmus:
  - Drücke die Leertaste, um die Suche zu starten.
  - Der Algorithmus sucht den kürzesten Weg vom Start zum Ziel und zeigt dabei:
    - Welche Felder schon besucht wurden (blau)
    - Welche Felder noch geprüft werden (gelb)
    - Den gefundenen Pfad (grün)
    - Start/Ziel (rot)
- Weitere Funktionen:
  - C: Grid zurücksetzen
  - R: Zufällige Hindernisse generieren
  - ESC: Programm beenden

Wie arbeitet der A*-Algorithmus?
- Er sucht immer das Feld mit den geringsten "Kosten" (Summe aus bisherigem Weg und geschätzter Entfernung zum Ziel).
- Er prüft alle Nachbarfelder, merkt sich, wie er dorthin gekommen ist, und wiederholt das, bis das Ziel erreicht ist.
- Am Ende wird der kürzeste Pfad farbig angezeigt.


"""

import pygame
import sys
import heapq
from typing import List, Tuple, Set, Dict

# Farben
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = None

    def __lt__(self, other):
        return self.f_cost < other.f_cost

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Berechnet die Manhattan-Distanz zwischen zwei Punkten."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[int, int]]:
    """Gibt die gültigen Nachbarn einer Position zurück."""
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Rechts, Unten, Links, Oben
    
    for dx, dy in directions:
        new_x, new_y = pos[0] + dx, pos[1] + dy
        
        if (0 <= new_x < len(grid) and 
            0 <= new_y < len(grid[0]) and 
            grid[new_x][new_y] != 1):
            neighbors.append((new_x, new_y))
    
    return neighbors

class AStarVisualizer:
    def __init__(self, grid_size: int = 20, cell_size: int = 30):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        self.sidebar_width = 250  # Breite für Steuerungshinweise
        
        # Initialisiere Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size + self.sidebar_width, self.window_size))
        pygame.display.set_caption("A* Algorithmus Visualisierung")
        self.clock = pygame.time.Clock()
        
        # Grid initialisieren (0 = frei, 1 = Hindernis)
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        
        # Start- und Endposition
        self.start_pos = None
        self.end_pos = None
        
        # Pfad und besuchte Knoten
        self.path = []
        self.visited = set()
        self.open_set = []
        
        # Maus-Status
        self.drawing = False
        self.erasing = False
        
        # Font für Text
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 20, bold=True)
        
        # Status
        self.status = "Bereit"
        self.status_color = BLACK
    
    def draw_sidebar(self):
        """Zeichnet die Seitenleiste mit Steuerungshinweisen."""
        sidebar_rect = pygame.Rect(self.window_size, 0, self.sidebar_width, self.window_size)
        pygame.draw.rect(self.screen, LIGHT_GRAY, sidebar_rect)
        
        # Titel
        title = self.title_font.render("Steuerung", True, BLACK)
        self.screen.blit(title, (self.window_size + 10, 20))
        
        # Steuerungshinweise
        controls = [
            "Linke Maustaste: Hindernisse zeichnen",
            "Rechte Maustaste: Hindernisse löschen",
            "Shift + Linksklick: Start setzen",
            "Strg + Linksklick: Ziel setzen",
            "Leertaste: Algorithmus starten",
            "C: Grid zurücksetzen",
            "R: Zufällige Hindernisse",
            "ESC: Beenden"
        ]
        
        y = 60
        for control in controls:
            text = self.font.render(control, True, BLACK)
            self.screen.blit(text, (self.window_size + 10, y))
            y += 30
        
        # Legende
        y += 20
        legend_title = self.title_font.render("Legende", True, BLACK)
        self.screen.blit(legend_title, (self.window_size + 10, y))
        y += 30
        
        for color, text in [(RED, "Start/End"), (BLUE, "Besucht"), 
                           (YELLOW, "Offene Liste"), (GREEN, "Pfad"),
                           (BLACK, "Hindernis")]:
            pygame.draw.rect(self.screen, color, (self.window_size + 10, y, 20, 20))
            text_surface = self.font.render(text, True, BLACK)
            self.screen.blit(text_surface, (self.window_size + 35, y))
            y += 25
        
        # Status
        y = self.window_size - 60
        status_text = self.font.render(f"Status: {self.status}", True, self.status_color)
        self.screen.blit(status_text, (self.window_size + 10, y))
    
    def draw_grid(self):
        """Zeichnet das Grid und alle Elemente."""
        self.screen.fill(WHITE)
        
        # Zeichne Grid-Linien
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, BLACK, 
                           (x * self.cell_size, 0), 
                           (x * self.cell_size, self.window_size))
            pygame.draw.line(self.screen, BLACK, 
                           (0, x * self.cell_size), 
                           (self.window_size, x * self.cell_size))
        
        # Zeichne Hindernisse
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x][y] == 1:
                    pygame.draw.rect(self.screen, BLACK,
                                   (y * self.cell_size, x * self.cell_size,
                                    self.cell_size, self.cell_size))
        
        # Zeichne besuchte Knoten
        for pos in self.visited:
            pygame.draw.rect(self.screen, BLUE,
                           (pos[1] * self.cell_size, pos[0] * self.cell_size,
                            self.cell_size, self.cell_size))
        
        # Zeichne offene Liste
        for node in self.open_set:
            pos = node.position
            pygame.draw.rect(self.screen, YELLOW,
                           (pos[1] * self.cell_size, pos[0] * self.cell_size,
                            self.cell_size, self.cell_size))
        
        # Zeichne Pfad
        for pos in self.path:
            pygame.draw.rect(self.screen, GREEN,
                           (pos[1] * self.cell_size, pos[0] * self.cell_size,
                            self.cell_size, self.cell_size))
        
        # Zeichne Start- und Endposition
        if self.start_pos:
            pygame.draw.rect(self.screen, RED,
                           (self.start_pos[1] * self.cell_size,
                            self.start_pos[0] * self.cell_size,
                            self.cell_size, self.cell_size))
        if self.end_pos:
            pygame.draw.rect(self.screen, RED,
                           (self.end_pos[1] * self.cell_size,
                            self.end_pos[0] * self.cell_size,
                            self.cell_size, self.cell_size))
        
        # Zeichne Seitenleiste
        self.draw_sidebar()
    
    def handle_events(self):
        """Behandelt Benutzereingaben."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if x < self.window_size:  # Nur im Grid-Bereich
                    grid_x = y // self.cell_size
                    grid_y = x // self.cell_size
                    
                    if event.button == 1:  # Linke Maustaste
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            self.start_pos = (grid_x, grid_y)
                            self.status = "Start gesetzt"
                            self.status_color = GREEN
                        elif pygame.key.get_mods() & pygame.KMOD_CTRL:
                            self.end_pos = (grid_x, grid_y)
                            self.status = "Ziel gesetzt"
                            self.status_color = GREEN
                        else:
                            self.drawing = True
                            self.grid[grid_x][grid_y] = 1
                            self.status = "Zeichne Hindernisse"
                            self.status_color = BLACK
                    
                    elif event.button == 3:  # Rechte Maustaste
                        self.erasing = True
                        self.grid[grid_x][grid_y] = 0
                        self.status = "Lösche Hindernisse"
                        self.status_color = BLACK
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self.drawing = False
                self.erasing = False
                if self.status in ["Zeichne Hindernisse", "Lösche Hindernisse"]:
                    self.status = "Bereit"
                    self.status_color = BLACK
            
            elif event.type == pygame.MOUSEMOTION:
                if self.drawing or self.erasing:
                    x, y = event.pos
                    if x < self.window_size:  # Nur im Grid-Bereich
                        grid_x = y // self.cell_size
                        grid_y = x // self.cell_size
                        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                            self.grid[grid_x][grid_y] = 1 if self.drawing else 0
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.start_pos and self.end_pos:
                    self.status = "Suche Pfad..."
                    self.status_color = BLUE
                    self.run_astar()
                elif event.key == pygame.K_c:
                    self.clear_grid()
                    self.status = "Grid zurückgesetzt"
                    self.status_color = GREEN
                elif event.key == pygame.K_r:
                    self.generate_random_obstacles()
                    self.status = "Zufällige Hindernisse generiert"
                    self.status_color = GREEN
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
    
    def generate_random_obstacles(self):
        """Generiert zufällige Hindernisse im Grid."""
        import random
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if random.random() < 0.3:  # 30% Wahrscheinlichkeit für Hindernisse
                    self.grid[x][y] = 1
    
    def clear_grid(self):
        """Setzt das Grid zurück."""
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.start_pos = None
        self.end_pos = None
        self.path = []
        self.visited = set()
        self.open_set = []
    
    def run_astar(self):
        """Führt den A* Algorithmus aus."""
        if not (self.start_pos and self.end_pos):
            return
        
        # Initialisiere Listen
        self.open_set = []
        self.visited = set()
        self.path = []
        
        # Erstelle Start- und Endknoten
        start_node = Node(self.start_pos, 0, heuristic(self.start_pos, self.end_pos))
        end_node = Node(self.end_pos)
        
        # Füge Startknoten zur offenen Liste hinzu
        heapq.heappush(self.open_set, start_node)
        node_dict = {self.start_pos: start_node}
        
        while self.open_set:
            # Hole Knoten mit niedrigsten f-Kosten
            current = heapq.heappop(self.open_set)
            
            # Wenn wir das Ziel erreicht haben
            if current.position == self.end_pos:
                # Rekonstruiere Pfad
                while current:
                    self.path.append(current.position)
                    current = current.parent
                self.path.reverse()
                self.status = "Pfad gefunden!"
                self.status_color = GREEN
                return
            
            # Füge aktuellen Knoten zur geschlossenen Liste hinzu
            self.visited.add(current.position)
            
            # Prüfe alle Nachbarn
            for neighbor_pos in get_neighbors(current.position, self.grid):
                if neighbor_pos in self.visited:
                    continue
                
                # Berechne neue g-Kosten
                new_g_cost = current.g_cost + 1
                
                # Wenn der Nachbar noch nicht in der offenen Liste ist oder
                # wenn wir einen besseren Pfad gefunden haben
                if neighbor_pos not in node_dict or new_g_cost < node_dict[neighbor_pos].g_cost:
                    neighbor = Node(
                        neighbor_pos,
                        new_g_cost,
                        heuristic(neighbor_pos, self.end_pos)
                    )
                    neighbor.parent = current
                    node_dict[neighbor_pos] = neighbor
                    heapq.heappush(self.open_set, neighbor)
            
            # Aktualisiere die Visualisierung
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(30)  # Begrenze die Framerate
        
        self.status = "Kein Pfad gefunden!"
        self.status_color = RED
    
    def run(self):
        """Hauptschleife der Visualisierung."""
        while True:
            self.handle_events()
            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    visualizer = AStarVisualizer()
    visualizer.run() 