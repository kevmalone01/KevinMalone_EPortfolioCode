"""
Einfacher k-Nearest Neighbors (kNN) Algorithmus
=============================================

Dieses Programm implementiert den k-Nearest Neighbors Algorithmus, einen der
einfachsten und intuitivsten Algorithmen des maschinellen Lernens. Der Algorithmus
klassifiziert neue Datenpunkte basierend auf den k nächsten Nachbarn im
Trainingsdatensatz.

Funktionsweise:
-------------
1. Berechnung der Distanz zwischen dem neuen Datenpunkt und allen Trainingsdaten
2. Auswahl der k nächsten Nachbarn
3. Mehrheitsentscheidung der k Nachbarn für die Klassifizierung

In diesem Beispiel:
- Wir klassifizieren Blumen in zwei Kategorien basierend auf zwei Merkmalen
- Die Merkmale sind: Blütenblattlänge und Blütenblattbreite
- Die Klassen sind: 0 (Klasse A) und 1 (Klasse B)
- k = 3 (wir betrachten die 3 nächsten Nachbarn)

Programmausgabe:
-------------
1. Trainingsdaten werden angezeigt
2. Ein neuer Datenpunkt wird klassifiziert
3. Die k nächsten Nachbarn werden angezeigt
4. Die finale Klassifizierung wird ausgegeben
"""

import numpy as np
from collections import Counter

class SimpleKNN:
    def __init__(self, k=3):
        """
        Initialisiert den kNN-Klassifizierer.
        
        Parameter:
        - k: Anzahl der zu betrachtenden nächsten Nachbarn
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Speichert die Trainingsdaten.
        
        Parameter:
        - X: Trainingsdaten (Merkmale)
        - y: Trainingslabels (Klassen)
        """
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        """
        Berechnet die euklidische Distanz zwischen zwei Datenpunkten.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Klassifiziert neue Datenpunkte.
        
        Parameter:
        - X: Neue Datenpunkte zur Klassifizierung
        
        Rückgabe:
        - Vorhergesagte Klassen für die neuen Datenpunkte
        """
        predictions = []
        
        for x in X:
            # Berechne Distanzen zu allen Trainingsdaten
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # Finde die k nächsten Nachbarn
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            # Mehrheitsentscheidung
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return predictions

# Beispiel: Blumenklassifizierung
if __name__ == "__main__":
    # Trainingsdaten erstellen
    # Format: [Blütenblattlänge, Blütenblattbreite]
    X_train = np.array([
        [1.0, 0.5],  # Klasse 0
        [1.2, 0.6],  # Klasse 0
        [1.4, 0.7],  # Klasse 0
        [4.0, 1.5],  # Klasse 1
        [4.2, 1.6],  # Klasse 1
        [4.4, 1.7]   # Klasse 1
    ])
    
    # Klassenlabels (0 oder 1)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    # kNN-Klassifizierer erstellen und trainieren
    knn = SimpleKNN(k=3)
    knn.fit(X_train, y_train)
    
    # Trainingsdaten anzeigen
    print("Trainingsdaten:")
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        print(f"Datenpunkt {i+1}: Merkmale = {x}, Klasse = {y}")
    
    # Neuen Datenpunkt klassifizieren
    new_point = np.array([[2.0, 1.0]])
    prediction = knn.predict(new_point)
    
    print("\nKlassifizierung eines neuen Datenpunkts:")
    print(f"Neuer Datenpunkt: Merkmale = {new_point[0]}")
    print(f"Vorhergesagte Klasse: {prediction[0]}")
    
    # Zeige die k nächsten Nachbarn
    distances = [knn.euclidean_distance(new_point[0], x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:knn.k]
    
    print("\nDie 3 nächsten Nachbarn:")
    for i, idx in enumerate(k_indices):
        print(f"Nachbar {i+1}: Merkmale = {X_train[idx]}, Klasse = {y_train[idx]}, "
              f"Distanz = {distances[idx]:.2f}") 