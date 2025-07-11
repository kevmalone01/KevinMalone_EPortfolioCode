# 2. Problemlösen als Suchaufgabe 

Viele Aufgabenstellungen in der Künstlichen Intelligenz lassen sich als **Suchprobleme** modellieren. Dabei geht es nicht darum, eine Lösung direkt zu berechnen, sondern sie durch das systematische **Durchsuchen eines Zustandsraums** zu finden. Dieses Vorgehen ist besonders relevant für Pfadplanung, Spielstrategien oder Entscheidungsfindung in komplexen Umgebungen.

---

## 2.1 Grundkonzept

Ein Suchproblem besteht typischerweise aus folgenden Komponenten:

* einem **Startzustand**,
* einem **Zielzustand** (oder Zieltest),
* einer Menge von **Operatoren** (mögliche Aktionen),
* und optional einer **Kostenfunktion**, die Lösungsqualität bewertet.

Ein Beispiel ist ein Agent in einem Labyrinth: Er startet an einem Punkt und soll einen Zielpunkt finden. Der Zustandsraum ist die Karte, die Operatoren sind Bewegungen wie "gehe nach oben", und das Ziel ist erreicht, wenn der Zielzustand betreten wird.

Man unterscheidet dabei zwischen:

* **Zustandsraum**: die Menge aller möglichen Zustände
* **Suchbaum**: alle tatsächlich durchlaufenen Zustandsfolgen während der Suche


![image](https://github.com/user-attachments/assets/82a56535-b514-4dad-9b2d-7ccf3ef95ba0)


*Abbildung 3 Entscheidungsbaum mit Zuständen, Aktionen und Zeit (Quelle: Gabler Wirtschaftslexikon 2024)*

### Suchverfahren

Es gibt zwei grundlegende Kategorien:

* **Uninformierte Suche**: keine Kenntnis über die Struktur des Problems (z. B. Tiefensuche, Breitensuche)
* **Informierte Suche**: nutzt Problemwissen, z. B. durch Heuristiken (z. B. A\*-Algorithmus)

Die Wahl des Verfahrens hängt vom Problem, den Ressourcen (Zeit/Speicher) und der Zielsetzung ab.

---

## 2.2 Navigation & Pfadplanung

Ein zentrales Anwendungsfeld ist die **Navigation**. Hier möchte ein Agent (z. B. Roboter, Spielfigur) den effizientesten Pfad von einem Startpunkt zum Ziel finden.

### Suche ohne Karte: Bug-Algorithmen

Wenn keine Karte vorhanden ist, kommt z. B. ein Bug-Algorithmus zum Einsatz:

* Der Agent bewegt sich direkt auf das Ziel zu
* Bei Kollision mit einem Hindernis folgt er dessen Rand (z. B. im Uhrzeigersinn)
* Sobald eine bessere Richtung möglich ist, verlässt er die Kante und setzt fort

![image](https://github.com/user-attachments/assets/bedd730d-c859-4ebd-9a93-559affa56401)

*Abbildung 4 Veranschaulichung des Bug-Algorithmus in Aktion (Quelle: ResearchGate-Trajectory of dist-bug algorithm)*

>  **Beispiel (siehe Abbildung 4)**: 
Ein mobiler Agent (z. B. ein Roboter) bewegt sich zunächst geradlinig auf das Ziel zu (gestrichelte Linie). Nach Kollision mit einem Hindernis folgt er dessen Rand (rote Trajektorie), bis er wieder in Richtung Ziel navigieren kann.


### Suche mit Karte: Graphbasierte Verfahren

Bei bekannter Umgebung kann man den Raum als **Graph** modellieren:

* **Knoten** = Orte oder Zustände
* **Kanten** = mögliche Übergänge mit optionalen Kosten

Typische Suchverfahren:

| Verfahren              | Beschreibung                   | Eigenschaften                                     |
| ---------------------- | ------------------------------ | ------------------------------------------------- |
| **Breitensuche (BFS)** | Ebene für Ebene, systematisch  | Vollständig, langsam, optimal bei gleichen Kosten |
| **Tiefensuche (DFS)**  | Tiefe zuerst, rekursiv         | Speicherschonend, nicht optimal                   |
| **Uniform Cost**       | Erweiterung von BFS mit Kosten | Optimal bei korrekten Kostenangaben               |
| **A**\*                | Bewertung: g(n) + h(n)         | Optimal bei zulässiger Heuristik                  |

> **Formel (A\*)**: f(n) = g(n) + h(n)
>
> * g(n): bisherige Kosten
> * h(n): geschätzte Restkosten zum Ziel

### Beispiel (Python): Pfad in 5×5-Matrix

```python
start = (0, 0)
goal = (4, 4)
grid = [
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0]
]
```

>  Ausgabe: (0,0) ➔ (0,1) ➔ (0,2) ➔ (1,2) ➔ … ➔ (4,4)

### Bewertungskriterien für Suchverfahren

* **Korrektheit**: Liefert nur valide Lösungen
* **Vollständigkeit**: Findet eine Lösung, wenn vorhanden
* **Optimalität**: Beste Lösung bezüglich Kosten
* **Effizienz**: Zeit- und Speicherverbrauch

---

## 2.3 Spielsuche (Zwei-Personen-Spiele)

Bei Spielen wie **Tic-Tac-Toe, Schach oder 4-Gewinnt** wechseln sich zwei Spieler ab. Ziel ist es, einen optimalen Zug zu wählen. Diese Probleme werden meist als **Nullsummenspiele** modelliert: Der Gewinn des einen ist der Verlust des anderen.

### MinMax-Algorithmus

* Basiert auf einem Spielbaum
* MAX will den Nutzen maximieren
* MIN will ihn minimieren

Die Bewertungen propagieren von den Blättern zur Wurzel zurück. Das Modell wählt den Pfad, der den besten garantierten Wert liefert.

> Beispiel: KI für TicTacToe simuliert alle Zugfolgen und wählt den besten aus.

### Alpha-Beta-Pruning

Optimierung für MinMax: Unnötige Pfade werden **abgeschnitten**, wenn klar ist, dass sie schlechter als bekannte Alternativen sind. Dadurch wird der Suchraum drastisch reduziert.

### Monte Carlo Tree Search (MCTS)

Anstelle vollständiger Berechnung simuliert MCTS viele **zufällige Spielverläufe (Rollouts)**, um Pfade statistisch zu bewerten. Dies eignet sich besonders bei komplexen Spielen wie **Go**.

---

## Fazit

Suchprobleme sind ein zentrales Konzept der KI. Ob Navigation oder Spielstrategie: Durch intelligente Auswahl und Bewertung von Pfaden können auch in großen Räumen effiziente Entscheidungen getroffen werden. Heuristiken, Pruning und Simulation sind dabei entscheidende Werkzeuge.

---
##  Quellen

* Russell & Norvig (2021): Artificial Intelligence: A Modern Approach. 4th ed., Pearson.

* GeeksforGeeks (2023): Problem Solving in AI. https://www.geeksforgeeks.org/problem-solving-in-artificial-intelligence/

* Gabler Wirtschaftslexikon (2024): Entscheidungsbaum. https://wirtschaftslexikon.gabler.de/definition/entscheidungsbaum-35225

* ResearchGate (2018): Trajectory of dist-bug algorithm. https://www.researchgate.net/figure/Trajectory-of-dist-bug-algorithm_fig1_324562768

* GeeksforGeeks (2023): Pathfinding Algorithms in AI. https://www.geeksforgeeks.org/search-algorithms-in-ai/

* Computerphile (YouTube): A Pathfinding Algorithm*. https://youtu.be/v-pSdigfwM8?si=0vp3E6KQwufdvdme

* Eigene Implementierung und Beispiele basierend auf KI-Übungsblatt „Wegsuche in Graphen“ (ki_ueb210_Path-2)

* Wikipedia (2025): Monte Carlo Tree Search. https://de.wikipedia.org/wiki/Monte-Carlo-Tree-Search

* Eigene Implementierung des MinMax-Algorithmus in Python zur Anwendung auf TicTacToe (TicTacToe.py)



