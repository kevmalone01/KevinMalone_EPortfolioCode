"""
Einfaches Neuronales Netzwerk mit Backpropagation
===============================================

Dieses Programm implementiert ein einfaches neuronales Netzwerk, das die grundlegenden
Konzepte des maschinellen Lernens demonstriert. Es verwendet die Backpropagation-Methode,
um das XOR-Problem zu lösen.

Architektur:
-----------
- Eingabeschicht: 2 Neuronen
- Versteckte Schicht: 2 Neuronen
- Ausgabeschicht: 1 Neuron

Funktionsweise:
-------------
1. Initialisierung:
   - Zufällige Gewichte und Bias-Terme werden erstellt
   - Jede Verbindung zwischen Neuronen hat ein eigenes Gewicht

2. Forward Propagation:
   - Eingabedaten werden durch das Netzwerk geleitet
   - Jede Schicht berechnet ihre Ausgabe mit der Sigmoid-Funktion
   - Zwischenergebnisse werden für Backpropagation gespeichert

3. Backpropagation:
   - Fehler zwischen Vorhersage und tatsächlichem Wert wird berechnet
   - Fehler wird rückwärts durch das Netzwerk propagiert
   - Gewichte werden basierend auf dem Fehler angepasst

4. Training:
   - 10.000 Epochen werden durchgeführt
   - In jeder Epoche: Forward Propagation → Backpropagation
   - Verlust wird alle 1000 Epochen ausgegeben

XOR-Problem:
----------
Das Programm demonstriert das Lernen am XOR-Problem:
Eingabe    Ausgabe
0 0    →   0
0 1    →   1
1 0    →   1
1 1    →   0

Dies ist ein klassisches Problem, das ein einfaches Perzeptron nicht lösen kann,
aber mit einem mehrschichtigen Netzwerk lösbar ist.

Programmausgabe:
-------------
1. Trainingsphase:
   - "Starte Training..." wird angezeigt
   - Alle 1000 Epochen wird der aktuelle Verlust (Loss) ausgegeben
   - Format: "Epoche X, Verlust: Y"
   - Der Verlust sollte mit der Zeit kleiner werden

2. Testphase:
   - "Teste das trainierte Netzwerk:" wird angezeigt
   - Für jede XOR-Eingabe wird ausgegeben:
     * Die Eingabewerte
     * Die erwartete Ausgabe
     * Die tatsächliche Vorhersage des Netzwerks (auf 4 Dezimalstellen gerundet)
   - Format: "Eingabe: [X Y], Erwartete Ausgabe: Z, Vorhersage: W"


"""

import numpy as np

class SimpleNeuralNetwork:
    """
    Einfaches neuronales Netzwerk mit Backpropagation.

    """
    
    def __init__(self):
        """
        Initialisiert das neuronale Netzwerk mit zufälligen Gewichten und Biases.
        - weights1: Verbindungen zwischen Eingabe- und versteckter Schicht (2x2 Matrix)
        - weights2: Verbindungen zwischen versteckter und Ausgabeschicht (2x1 Matrix)
        - bias1: Bias-Terme für die versteckte Schicht (2 Vektoren)
        - bias2: Bias-Term für die Ausgabeschicht (1 Vektor)
        """
        self.weights1 = np.random.randn(2, 2)  # Gewichte Eingabe → versteckte Schicht
        self.weights2 = np.random.randn(2, 1)  # Gewichte versteckte Schicht → Ausgabe
        self.bias1 = np.random.randn(2)        # Bias für versteckte Schicht
        self.bias2 = np.random.randn(1)        # Bias für Ausgabeschicht

    def sigmoid(self, x):
        """
        Sigmoid-Aktivierungsfunktion.
        Komprimiert die Eingabe auf einen Wert zwischen 0 und 1.
        Formel: f(x) = 1 / (1 + e^(-x))
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Ableitung der Sigmoid-Funktion.
        Wird für die Backpropagation benötigt.
        Formel: f'(x) = f(x) * (1 - f(x))
        """
        return x * (1 - x)

    def forward(self, X):
        """
        Führt den Vorwärtsdurchlauf durch das Netzwerk durch.
        
        Parameter:
        - X: Eingabedaten (Matrix)
        
        Prozess:
        1. Berechnet die Ausgabe der versteckten Schicht
        2. Berechnet die finale Ausgabe
        3. Speichert Zwischenergebnisse für Backpropagation
        """
        # Berechne Ausgabe der versteckten Schicht
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        # Berechne finale Ausgabe
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, output, learning_rate=0.1):
        """
        Führt die Backpropagation durch und aktualisiert die Gewichte.
        
        Parameter:
        - X: Eingabedaten
        - y: Erwartete Ausgabe
        - output: Tatsächliche Ausgabe des Netzwerks
        - learning_rate: Lernrate für die Gewichtsaktualisierung
        """
        # Berechne den Fehler zwischen Vorhersage und tatsächlichem Wert
        error = y - output
        
        # Berechne Gradient für die Ausgabeschicht
        d_predicted_output = error * self.sigmoid_derivative(output)
        
        # Berechne Fehler in der versteckten Schicht
        error_hidden = d_predicted_output.dot(self.weights2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden)
        
        # Aktualisiere die Gewichte
        self.weights2 += self.hidden.T.dot(d_predicted_output) * learning_rate
        self.weights1 += X.T.dot(d_hidden) * learning_rate
        
        # Aktualisiere die Bias-Terme
        self.bias2 += np.sum(d_predicted_output, axis=0) * learning_rate
        self.bias1 += np.sum(d_hidden, axis=0) * learning_rate

    def train(self, X, y, epochs=10000):
        """
        Trainiert das neuronale Netzwerk.
        
        Parameter:
        - X: Trainingsdaten
        - y: Erwartete Ausgaben
        - epochs: Anzahl der Trainingsdurchläufe
        """
        for epoch in range(epochs):
            # Vorwärtsdurchlauf
            output = self.forward(X)
            
            # Backpropagation
            self.backward(X, y, output)
            
            # Zeige Fortschritt alle 1000 Epochen
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoche {epoch}, Verlust: {loss}')

# Beispiel: XOR-Problem
if __name__ == "__main__":
    # Trainingsdaten für das XOR-Problem
    X = np.array([[0, 0],  # Eingabe 1
                  [0, 1],  # Eingabe 2
                  [1, 0],  # Eingabe 3
                  [1, 1]]) # Eingabe 4
    
    y = np.array([[0],     # Erwartete Ausgabe für Eingabe 1
                  [1],     # Erwartete Ausgabe für Eingabe 2
                  [1],     # Erwartete Ausgabe für Eingabe 3
                  [0]])    # Erwartete Ausgabe für Eingabe 4

    # Erstelle und trainiere das Netzwerk
    print("Starte Training...")
    nn = SimpleNeuralNetwork()
    nn.train(X, y)
    
    # Teste das trainierte Netzwerk
    print("\nTeste das trainierte Netzwerk:")
    for i in range(len(X)):
        prediction = nn.forward(X[i:i+1])
        print(f"Eingabe: {X[i]}, Erwartete Ausgabe: {y[i][0]}, Vorhersage: {prediction[0][0]:.4f}") 