"""
K-Means Clustering im Terminal
-----------------------------

Dieses Programm zeigt, wie das K-Means-Verfahren zur Gruppierung von Datenpunkten funktioniert – ganz ohne externe Pakete.

- Die Datenpunkte werden in k Cluster eingeteilt (hier: k=3).
- Nach jeder Iteration werden die Cluster und ihre Zentren im Terminal ausgegeben.
- Das Programm stoppt automatisch, wenn sich die Zentren nicht mehr verändern.

Bedienung:
- Einfach ausführen: python kmeans_terminal.py
- Die Entwicklung der Cluster kann im Terminal Schritt für Schritt verfolgt werden.

Ideal, um das Grundprinzip von K-Means und Clustering zu verstehen.

K-Means Clustering – Minimalbeispiel 
---------------------------------------------------------

Dieses Programm demonstriert das K-Means-Verfahren zur Gruppierung (Clustering) von Datenpunkten im 2D-Raum.
Es ist komplett in Standard-Python geschrieben und benötigt keine Zusatzpakete.

Was macht das Programm?
-----------------------
- Es erzeugt eine kleine Menge von 2D-Datenpunkten, die nicht klar in Gruppen liegen.
- Es teilt die Punkte in k Cluster (hier: k=3) ein.
- Die Clusterzuordnung und die Zentren werden nach jeder Iteration im Terminal ausgegeben.
- Das Programm stoppt automatisch, wenn sich die Clusterzentren nicht mehr verändern (Stopkriterium).

Ablauf des K-Means-Algorithmus:
-------------------------------
1. Wähle k Startzentren zufällig aus den Datenpunkten.
2. Wiederhole:
   a) Weise jeden Punkt dem nächstgelegenen Zentrum zu (nach euklidischer Distanz).
   b) Berechne für jedes Cluster das neue Zentrum (Mittelwert aller Punkte im Cluster).
   c) Stoppe, wenn sich die Zentren nicht mehr ändern.


Hinweis:
--------
- Bei klar getrennten Daten findet K-Means die Lösung sehr schnell.
- Bei gemischten/verteilten Daten sieht man, wie die Cluster und Zentren sich über mehrere Iterationen verändern.
- Die Ausgabe im Terminal zeigt die Entwicklung Schritt für Schritt.


"""

import random

# Beispiel-Daten: 2D-Punkte, diesmal gemischt verteilt
data = [
    [1, 2], [2, 1], [3, 2], [8, 8], [9, 8], [8, 9],
    [5, 5], [6, 5], [5, 6], [7, 2], [2, 7], [6, 1]
]

k = 3  # Anzahl der Cluster

# Zufällige Startzentren wählen
centers = random.sample(data, k)

for iteration in range(1, 101):  # Maximal 100 Iterationen als Sicherheit
    # Schritt 1: Zuweisung der Punkte zu den nächsten Zentren
    clusters = [[] for _ in range(k)]
    for point in data:
        distances = [((point[0]-cx)**2 + (point[1]-cy)**2)**0.5 for cx, cy in centers]
        cluster_idx = distances.index(min(distances))
        clusters[cluster_idx].append(point)
    # Schritt 2: Zentren neu berechnen
    new_centers = []
    for cluster in clusters:
        if cluster:
            x_mean = sum(p[0] for p in cluster) / len(cluster)
            y_mean = sum(p[1] for p in cluster) / len(cluster)
            new_centers.append([x_mean, y_mean])
        else:
            new_centers.append(random.choice(data))
    print(f"Iteration {iteration}:")
    for idx, cluster in enumerate(clusters):
        print(f"  Cluster {idx+1}: {cluster}")
    print(f"  Zentren: {new_centers}\n")
    # Stopkriterium: Wenn sich die Zentren nicht mehr ändern, abbrechen
    if new_centers == centers:
        print("Stopp: Zentren haben sich nicht mehr verändert.")
        break
    centers = new_centers

print("Finale Cluster-Zentren:", centers) 