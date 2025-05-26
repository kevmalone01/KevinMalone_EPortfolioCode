# 3. Ansätze zum Maschinellen Lernen

## 3.1 Grundlagen & Lernarten des maschinellen Lernens

Lernen ist ein grundlegender Bestandteil menschlicher Entwicklung – sei es durch formelle Bildung oder informelle Erfahrungen im Alltag. In der Informatik bildet dieses Prinzip die Basis des maschinellen Lernens (ML): Computer sollen aus Beispielen lernen, um Aufgaben besser zu lösen, ohne explizit programmiert zu sein.

Maschinelles Lernen ist ein Teilgebiet der Künstlichen Intelligenz. Ziel ist es, aus vorhandenen Daten ein Modell zu erstellen, das Muster erkennt und neue, unbekannte Daten korrekt verarbeitet. Dabei werden Zusammenhänge, Strukturen oder Entscheidungsregeln gelernt und anschließend eigenständig angewendet.

### Überblick über die Lernarten:

- **Überwachtes Lernen (Supervised Learning):**  
  Jedes Trainingsbeispiel enthält ein Zielattribut (Label). Ziel ist es, Modelle zur Klassifikation oder Regression zu trainieren.  
  *Beispiel:* Klassifikation von E-Mails als „Spam“ oder „Nicht-Spam“.

- **Unüberwachtes Lernen (Unsupervised Learning):**  
  Es gibt keine Zielwerte. Ziel ist es, Strukturen oder Muster in Daten zu erkennen – z. B. durch Clustering.  
  *Beispiel:* Kundensegmentierung im Marketing.

- **Bestärkendes Lernen (Reinforcement Learning):**  
  Ein Agent lernt durch Interaktion mit seiner Umwelt und erhält Belohnungen oder Strafen. Ziel ist das Lernen einer optimalen Strategie.  
  *Beispiel:* Autonomes Fahren oder Spielstrategien.

- **Self-Supervised / Semi-Supervised Learning:**  
  Kombinationen aus überwachten und unüberwachten Methoden, oft mit automatisch erzeugten Labels.

---

## 3.2 Klassifikationsaufgabe & Daten

Ein häufiges Anwendungsfeld im überwachten Lernen ist die Klassifikation. Dabei wird gelernt, neue Datenpunkte in Klassen einzuordnen – basierend auf zuvor gelernten Beispielen.

### Aufbau eines Trainingsdatensatzes:

Ein Datensatz besteht aus mehreren **Merkmalen (Features)** und einem **Zielattribut (Label)**. Typische Merkmalsarten sind:

- numerisch: z. B. Alter, Einkommen  
- kategorisch: z. B. Geschlecht, Farbe  
- ordinal: z. B. Schulnoten, Bewertungen

**Beispielhafte Datenpunktstruktur:**  
`[Alter = 45, Einkommen = hoch, Kreditwürdig = ja]`

Ziel ist es, mit Hilfe solcher strukturierter Trainingsdaten eine Vorhersagefunktion zu lernen.

### Datenaufbereitung (Preprocessing):

- Bereinigung fehlender oder fehlerhafter Werte  
- Normalisierung numerischer Merkmale  
- One-Hot-Encoding für kategoriale Merkmale  
- Aufteilung in Trainings- und Testdaten (z. B. 80 / 20)

> **Merksatz:** *Ein Modell ist nur so gut wie die Daten, auf denen es trainiert wurde.*

---

## 3.3 Wichtige Lernverfahren

Die Wahl des geeigneten Lernverfahrens hängt von der Datenlage, Zielsetzung und Modelltransparenz ab:

| Verfahren         | Typ            | Vorteile                        | Nachteile                        |
|------------------|----------------|----------------------------------|----------------------------------|
| Entscheidungsbaum | symbolisch     | gut interpretierbar              | überanpassungsanfällig           |
| Naive Bayes       | probabilistisch| schnell, robust                 | Unabhängigkeitsannahme unrealistisch |
| SVM (Support Vector Machine) | geometrisch | hohe Genauigkeit               | schwer interpretierbar           |
| Künstliche Neuronale Netze | subsymbolisch | hohe Modellleistung (z. B. bei Bildern, Sprache) | benötigt viel Daten & Rechenleistung |

> In der Praxis werden oft verschiedene Modelle getestet (z. B. mit Kreuzvalidierung), um das beste Verfahren für eine konkrete Aufgabe zu identifizieren.

---

**Quellen:**

1. Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.  
2. Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.).  
3. Wikipedia (2025): [Maschinelles Lernen](https://de.wikipedia.org/wiki/Maschinelles_Lernen)  
4. Datasolut (2024): [Was ist Machine Learning?](https://datasolut.com/was-ist-machine-learning/)  
5. GeeksforGeeks: [K-Means Clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/)  
6. DatabaseCamp: [SVM erklärt](https://databasecamp.de/ki/support-vector-machine-svm)  

