"""
Q-Learning Minimal Beispiel
===========================

Dieses Programm demonstriert Q-Learning, eine grundlegende Methode des Reinforcement Learning,
in ihrer einfachsten Form. Es zeigt, wie ein Agent lernt, sich in einer linearen Umgebung
von einem Startpunkt zu einem Ziel zu bewegen.

Die Umgebung:
------------
- 5 Zustände (0 bis 4)
- 2 mögliche Aktionen: Vorwärts (1) oder Rückwärts (0)
- Belohnung: 1 im Zielzustand (4), 0 sonst
- Der Agent startet immer bei 0 und soll zu 4 kommen

Wie es funktioniert:
------------------
1. Der Agent hat eine Q-Tabelle, die für jeden Zustand und jede Aktion einen Wert speichert
2. In jedem Schritt:
   - Wählt der Agent eine Aktion (manchmal zufällig, meist die beste)
   - Führt die Aktion aus und bekommt eine Belohnung
   - Aktualisiert seine Q-Tabelle basierend auf der Erfahrung
3. Nach vielen Episoden hat der Agent gelernt, den optimalen Pfad zu finden

Die Parameter:
-------------
- alpha (0.1): Lernrate - wie stark der Agent aus jeder Erfahrung lernt
- gamma (0.9): Diskontfaktor - wie wichtig zukünftige Belohnungen sind
- epsilon (0.2): Zufallsrate - wie oft der Agent neue Aktionen ausprobiert
- episodes (200): Anzahl der Übungsepisoden

Ausgabe:
--------
- Die finale Q-Tabelle zeigt die gelernten Werte für jede Aktion in jedem Zustand
- Der optimale Pfad zeigt den besten Weg vom Start zum Ziel

Beispiel:
---------
Zustand 0: [0.61, 0.73]  # [Rückwärts, Vorwärts]
Zustand 1: [0.57, 0.81]
Zustand 2: [0.64, 0.90]
Zustand 3: [0.70, 1.00]
Zustand 4: [0.00, 0.00]

Optimaler Pfad: 0 -> 1 -> 2 -> 3 -> 4

Die höheren Werte in der zweiten Spalte zeigen, dass "Vorwärts" in jedem Zustand
die bessere Aktion ist, und die Werte steigen, je näher man dem Ziel kommt.
"""

import random

# Parameter
num_states = 5    # Anzahl der Zustände (0 bis 4)
num_actions = 2   # Anzahl der Aktionen (0 = rückwärts, 1 = vorwärts)
alpha = 0.1       # Lernrate (wie schnell lernt der Agent?)
gamma = 0.9       # Diskontfaktor (wie wichtig sind zukünftige Belohnungen?)
epsilon = 0.2     # Zufallsrate (wie oft probiert der Agent zufällige Aktionen?)
episodes = 200    # Anzahl der Übungsepisoden

q_table = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]

def step(state, action):
    """Simuliert die Umgebung: gibt neuen Zustand und Belohnung zurück."""
    if action == 1:  # vorwärts
        next_state = min(state + 1, num_states - 1)
    else:            # rückwärts
        next_state = max(state - 1, 0)
    reward = 1 if next_state == num_states - 1 else 0
    return next_state, reward

for ep in range(episodes):
    state = 0
    for _ in range(20):  # max. 20 Schritte pro Episode
        # Aktion wählen (Epsilon-Greedy)
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            action = 0 if q_table[state][0] > q_table[state][1] else 1

        next_state, reward = step(state, action)
        # Q-Learning Update
        best_next = max(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])
        state = next_state
        if state == num_states - 1:
            break

# Zeige die gelernten Q-Werte und den optimalen Pfad
print("Q-Tabelle:")
for s in range(num_states):
    print(f"Zustand {s}: Rückwärts={q_table[s][0]:.2f}, Vorwärts={q_table[s][1]:.2f}")

print("\nOptimaler Pfad von 0 nach 4:")
state = 0
path = [state]
while state != num_states - 1:
    action = 0 if q_table[state][0] > q_table[state][1] else 1
    state, _ = step(state, action)
    path.append(state)
print(" -> ".join(map(str, path))) 