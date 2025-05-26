# 4.1 Unüberwachtes Lernen: Clustering und Strukturerkennung

Unüberwachtes Lernen – insbesondere Clustering – gehört zu den zentralen Aufgaben des Data Mining. Ziel ist es, unbekannte Muster, Gruppen oder Strukturen in großen Datenmengen zu erkennen – ohne dass Zielwerte oder Klassen vorgegeben sind. Das Vorgehen ist explorativ.

---

## Clustering (Klassenbildung)

Beim Clustering werden Datenobjekte so gruppiert, dass:

- **innerhalb eines Clusters** eine hohe Ähnlichkeit besteht
- **zwischen Clustern** deutliche Unterschiede erkennbar sind

### Typische Anwendungsbeispiele:

- Kundensegmentierung im Marketing  
- Gruppierung von Webseitenbesuchern  
- Einordnung von Aktien nach Risikoprofil

---

## Merkmalsarten & Ähnlichkeitsmaße

Die Eingabedaten können unterschiedliche Merkmalsarten enthalten:

- **Nominale Merkmale:** z. B. Farben, Produktnamen  
- **Ordinale Merkmale:** z. B. Bewertungen, Schulnoten  
- **Metrische Merkmale:** z. B. Preis, Größe, Gewicht

Je nach Merkmalsart kommen verschiedene **Distanzmaße** zum Einsatz, z. B.:

- Euklidische Distanz  
- Manhattan-Distanz  
- Jaccard-Ähnlichkeit (für binäre Merkmale)

---

## Clustering-Verfahren

### 1. Hierarchisches Clustering

- **Bottom-up (agglomerativ):** Startet mit einzelnen Objekten, die sukzessive zu Clustern zusammengeführt werden.
- **Darstellung:** meist als **Dendrogramm**.
- **Distanzstrategien zwischen Clustern:**
  - Single Linkage (nächstes Nachbarprinzip)
  - Complete Linkage
  - Centroid-Methode

### 2. k-Means-Verfahren

- Anzahl der Cluster **k** wird im Voraus festgelegt  
- Initiale **Clusterzentren** (Centroids) werden zufällig gewählt  
- Objekte werden dem **nächstgelegenen Zentrum** zugeordnet  
- Die **Zentren werden aktualisiert**, bis sich keine Änderung mehr ergibt

**Vorteile:**
- Sehr effizient und einfach implementierbar

**Nachteile:**
- Nur für konvexe Cluster geeignet  
- Ergebnis hängt stark von der Initialisierung ab

![image](https://github.com/user-attachments/assets/ed213297-7863-4fba-8890-98e3d633adff)

*Abbildung: Funktionsweise des k-Means-Algorithmus*

---

## Praktische Umsetzung

Clustering wird in vielen Tools unterstützt:

- **KNIME**: grafische Workflows, integrierte k-Means-Komponente  
- **WEKA**: bietet zahlreiche unüberwachte Lernverfahren zur Auswahl  
- Auch in Python (z. B. `scikit-learn`) oder R lässt sich Clustering sehr gut umsetzen

In praxisnahen Übungen werden Verfahren wie:

- **k-Means**
- **k-Nächste-Nachbarn (kNN)** zur Clustervalidierung

erprobt und analysiert.

---

## Quellen

1. GeeksforGeeks – *K-Means Clustering Introduction*  
   [https://www.geeksforgeeks.org/k-means-clustering-introduction/](https://www.geeksforgeeks.org/k-means-clustering-introduction/)

2. DatabaseCamp – *Support Vector Machine (SVM)*  
   [https://databasecamp.de/ki/support-vector-machine-svm](https://databasecamp.de/ki/support-vector-machine-svm)

3. SpringerLink – *Klassifikation mit maschinellem Lernen*  
   Schmid, U.; Stojanovic, N. (2021): Einsatz von Entscheidungsbäumen zur Klassifikation in der KI  
   [https://link.springer.com/article/10.1007/s00287-021-01398-0](https://link.springer.com/article/10.1007/s00287-021-01398-0)

