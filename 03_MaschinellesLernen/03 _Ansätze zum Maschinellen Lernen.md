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

Ein zentrales Anwendungsfeld des überwachten Lernens ist die Klassifikation. Dabei geht es darum, neue Datenpunkte automatisch einer bestimmten Klasse zuzuordnen, basierend auf zuvor gelernten Beispielen. Die zugrunde liegende Idee: Wenn man einem Lernsystem ausreichend Beispiele zeigt – etwa E-Mails, die als Spam oder nicht Spam markiert sind –, soll es anschließend auch neue, unbekannte E-Mails korrekt
einordnen können.

### Aufbau eines Trainingsdatensatzes:

Damit ein Klassifikator lernen kann, benötigt er strukturierte Trainingsdaten. Dieser Datensatz besteht aus mehreren **Merkmalen (Features)** und einem **Zielattribut (Label)**. Typische Merkmalsarten sind:

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


## 3.3 Wichtige Lernverfahren

Maschinelles Lernen umfasst eine Vielzahl von Algorithmen, die jeweils auf unterschiedliche Problemstellungen, Datentypen und Anforderungen zugeschnitten sind. Im Folgenden werden vier grundlegende Lernverfahren vorgestellt, die in der Praxis besonders häufig eingesetzt werden: **Entscheidungsbäume**, **k-Nearest Neighbors (kNN)**, **Naive Bayes** und **Support Vector Machines (SVM)**.


### Entscheidungsbäume

Ein Entscheidungsbaum ist ein baumartiges Modell, das Entscheidungen durch das schrittweise Aufspalten von Merkmalen trifft. Jeder innere Knoten prüft ein Merkmal, jede Kante steht für eine mögliche Ausprägung, und jedes Blatt liefert eine Vorhersage (z. B. Klasse *A* oder *B*).

![Entscheidungsbaum – Quelle: SpringerLink – Kless et al. (2021)]

![image](https://github.com/user-attachments/assets/5c8700bd-7d4f-4f54-9525-8f6f000a2249)

*Beispielhafte Darstellung eines Entscheidungsbaums*

**Beispiel:**  
Ein Modell soll entscheiden, ob Sport im Freien stattfinden soll. Es prüft nacheinander die Merkmale wie „Wetterlage“, „Temperatur“ und „Regenwahrscheinlichkeit“ und trifft auf dieser Basis die Entscheidung „ja“ oder „nein“.

**Vorteile:**
- Intuitiv verständlich & gut visuell darstellbar
- Erklärt Entscheidungen transparent (White Box)

**Nachteile:**
- Gefahr der Überanpassung (Overfitting)
- Instabil bei kleinen Datenänderungen



### k-Nearest Neighbors (kNN)

kNN ist ein instanzbasiertes Verfahren, das keine explizite Modellierung durchführt. Stattdessen wird bei der Vorhersage eines neuen Datenpunkts geprüft, welche *k* Trainingspunkte im Merkmalsraum am nächsten liegen (z. B. über euklidische Distanz).

**Beispiel:**  
Ein System soll bestimmen, ob ein Patient erkältet ist. Dazu wird er mit früheren Patienten verglichen, deren Symptome bekannt sind. Stimmen mehrere überein (z. B. Fieber, Husten), entscheidet das Modell entsprechend der Mehrheit der Nachbarn.

**Vorteile:**
- Einfach umzusetzen
- Keine Trainingsphase notwendig

**Nachteile:**
- Langsame Vorhersage bei großen Datenmengen
- Sensibel gegenüber irrelevanten Merkmalen



### Naive Bayes

Naive Bayes ist ein probabilistisches Verfahren, das auf dem Satz von Bayes basiert. Es geht davon aus, dass die Merkmale eines Datenpunkts bedingungsunabhängig voneinander sind – eine Annahme, die in der Praxis oft verletzt wird, aber erstaunlich gut funktioniert.

**Beispiel:**  
Ein Spamfilter zählt, wie häufig bestimmte Wörter wie „Geld“, „Jetzt kaufen“ oder „Gratis“ in einer E-Mail vorkommen. Auf dieser Basis wird berechnet, wie wahrscheinlich es ist, dass die Mail Spam ist.

**Vorteile:**
- Sehr schnelle Trainings- und Vorhersagezeiten
- Gut geeignet für große Merkmalsräume (z. B. Textklassifikation)

**Nachteile:**
- Unabhängigkeitsannahme oft nicht realistisch
- Wahrscheinlichkeitsschätzungen sind nicht immer präzise



### Support Vector Machines (SVM)

SVMs versuchen, zwei Klassen durch eine optimale Trennlinie (Hyperplane) zu unterscheiden. Ziel ist es, die Grenze so zu legen, dass der Abstand zu den nächstliegenden Datenpunkten beider Klassen maximal ist. Dies erhöht die Generalisierbarkeit des Modells.

![image](https://github.com/user-attachments/assets/ab69280d-def9-4067-8d6e-b3b56e9a44a5)


*Trennung zweier Klassen durch eine SVM*

**Beispiel:**  
Ein SVM-Modell wird mit echten und betrügerischen Banktransaktionen trainiert. Die Methode lernt eine Grenze, die beide Klassen im Merkmalsraum möglichst gut trennt.

**Vorteile:**
- Sehr hohe Genauigkeit bei klar trennbaren Klassen
- Funktioniert auch bei komplexeren Grenzen durch Kerneltricks

**Nachteile:**
- Aufwendige Parametereinstellung (z. B. C, Kernel, Gamma)
- Eingeschränkte Interpretierbarkeit



> In der Praxis werden oft verschiedene Modelle ausprobiert und mit Validierungsmethoden (z. B. Kreuzvalidierung) verglichen, um die beste Leistung zu erzielen. Die Wahl hängt stark von Datenmenge, Ziel, Rechenzeit und Erklärbarkeit ab.


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
