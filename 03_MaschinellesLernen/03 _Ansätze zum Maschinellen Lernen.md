# 3. Ansätze zum Maschinellen Lernen

## 3.1 Grundlagen & Lernarten des maschinellen Lernens

Lernen ist ein grundlegender Bestandteil menschlicher Entwicklung – sei es durch formelle Bildung oder informelle Erfahrungen im Alltag. In der Informatik bildet dieses Prinzip die Basis des maschinellen Lernens (ML): Computer sollen aus Beispielen lernen, um Aufgaben besser zu lösen, ohne explizit programmiert zu sein.

![image](https://github.com/user-attachments/assets/700759bd-b973-42c1-a831-eefef44c2ec4)

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

### Datenaufbereitung (Preprocessing):

- Bereinigung fehlender oder fehlerhafter Werte  
- Normalisierung numerischer Merkmale  
- One-Hot-Encoding für kategoriale Merkmale  
- Aufteilung in Trainings- und Testdaten (z. B. 80 / 20)

> **Merksatz:** *Ein Modell ist nur so gut wie die Daten, auf denen es trainiert wurde.*


### Damit ein Modell überhaupt lernen kann, braucht es drei zentrale Bausteine:

### 1. **Hypothesenraum (Modellklasse)**

Der Hypothesenraum umfasst die Menge aller möglichen Formen der Funktion \( f \), die das Modell ausprobieren kann – z. B.:

- Lineare Funktionen  
- Entscheidungsbäume  
- Neuronale Netze  

Er legt also fest, welche Arten von Zusammenhängen das Modell grundsätzlich abbilden kann.

### 2. **Fehlerfunktion (Loss Function)**

Die Fehlerfunktion bewertet, wie stark die Vorhersagen des Modells von den tatsächlichen Werten abweichen. Sie gibt an, wie „schlecht“ eine bestimmte Hypothese \( f \) ist.

- **Bei Klassifikation:**  
  - **0/1-Verlust:** 0 bei korrekter, 1 bei falscher Klassifikation  
  - **Kreuzentropie (Cross Entropy):** misst die Abweichung von Wahrscheinlichkeiten

- **Bei Regression (Zahlenwerte als Ziel):**  
  - **MSE (Mean Squared Error):** mittlere quadratische Abweichung

**Beispiel:**  
Wenn ein Modell für eine E-Mail vorhersagt: \( f(x) = 0.9 \) (also 90 % Spam-Wahrscheinlichkeit), aber \( y = 0 \) ist (kein Spam), ist der Fehler hoch. Die Loss-Funktion weist auf eine notwendige Korrektur hin.

### 3. **Optimierungsverfahren**

Ein Optimierungsverfahren passt das Modell so an, dass der Fehler auf den Trainingsdaten möglichst klein wird. Es verändert die Modellparameter – z. B. die Gewichte eines neuronalen Netzes – schrittweise.

- **Beispiel:**  
  Der Algorithmus verändert das Gewicht \( w_1 \), wenn z. B. das Wort „Geld“ im Betreff zu oft (oder zu selten) zur Klassifikation „Spam“ führt.

Ziel ist es, im Fehlerraum das **globale Minimum** zu finden, also die Parameterkombination mit dem geringsten Gesamtfehler.

![image](https://github.com/user-attachments/assets/5fe1e851-0b72-44c0-a1c4-60955720f279)

*Abbildung: Fehlerlandschaft – das Ziel ist das tiefste Tal (globales Minimum)*

### Was bedeutet das in der Praxis?

Ein ML-Modell **lernt nicht durch Auswendiglernen**, sondern durch **systematisches Anpassen seiner Vorhersagefunktion**. Es wird darauf trainiert, bei bekannten Beispielen gut abzuschneiden – und soll danach auch neue, unbekannte Fälle möglichst richtig einordnen können.

**Beispiel:**  
Nach dem Training auf 1.000 markierten E-Mails soll das Modell auch eine neue, noch nie gesehene E-Mail möglichst korrekt als „Spam“ oder „Nicht-Spam“ klassifizieren.

---

## 3.3 Klassifikationsaufgabe & Daten

Ein häufiges Anwendungsfeld im überwachten Lernen ist die Klassifikation. Dabei wird gelernt, neue Datenpunkte in Klassen einzuordnen – basierend auf zuvor gelernten Beispielen.

### Aufbau eines Trainingsdatensatzes:

Ein Datensatz besteht aus mehreren **Merkmalen (Features)** und einem **Zielattribut (Label)**. Typische Merkmalsarten sind:

- numerisch: z. B. Alter, Einkommen  
- kategorisch: z. B. Geschlecht, Farbe  
- ordinal: z. B. Schulnoten, Bewertungen

**Beispielhafte Datenpunktstruktur:**  
`[Alter = 45, Einkommen = hoch, Kreditwürdig = ja]`

### Datenaufbereitung (Preprocessing):

- Bereinigung fehlender oder fehlerhafter Werte  
- Normalisierung numerischer Merkmale  
- One-Hot-Encoding für kategoriale Merkmale  
- Aufteilung in Trainings- und Testdaten (z. B. 80 / 20)

> **Merksatz:** *Ein Modell ist nur so gut wie die Daten, auf denen es trainiert wurde.*

---

## 3.4 Wichtige Lernverfahren

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
