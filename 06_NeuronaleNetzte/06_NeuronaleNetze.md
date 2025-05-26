# 6. Neuronale Netze

Neuronale Netze sind ein zentraler Bestandteil moderner KI-Systeme. Sie sind inspiriert vom menschlichen Gehirn, bestehen jedoch aus **künstlichen Neuronen**, die Informationen verarbeiten, weiterleiten und lernen, Muster in Daten zu erkennen.

---

## 6.1 Grundidee künstlicher Neuronen

Ein künstliches Neuron erhält mehrere Eingangswerte, multipliziert sie mit Gewichtungen, summiert die Ergebnisse auf und entscheidet über eine **Aktivierungsfunktion**, ob und wie stark es „anspringt“.

### Aufbau eines künstlichen Neurons (Perzeptron):

Das Perzeptron ist das älteste und zugleich einfachste Modell eines künstlichen Neurons. Es wurde Ende der 1950er Jahre von Frank Rosenblatt entwickelt und kann lineare Entscheidungsgrenzen erlernen. Ein Perzeptron besteht aus:

- **Eingaben:** \( x_1, x_2, ..., x_n \)  
- **Gewichte:** \( w_1, w_2, ..., w_n \)  
- **Berechnung:**  
 z =   ∑ wi xi + b (mit Bias b) wobei \( b \) der **Bias** ist

- **Aktivierung:** Einer Aktivierungsfunktion (z.B. ein Schwellenwert), die entscheidet, ob das Neuron „feuert“


Das Perzeptron lernt durch Anpassung der Gewichte, ob ein Datenpunkt zur Klasse 1 oder 0 gehört. Es kann jedoch nur linear separierbare Probleme lösen.

---

## Netzstruktur: Von der Eingabe zur Vorhersage

Ein einzelnes Neuron ist nicht ausreichend für komplexe Aufgaben. Erst durch das **Verknüpfen mehrerer Neuronen zu Netzwerken** entstehen leistungsfähige Strukturen:

### Typische Architektur eines neuronalen Netzes:

- **Eingabeschicht:** Nimmt Rohdaten auf (z. B. Pixel, Sensordaten)
- **Versteckte Schichten (Hidden Layers):** Transformieren und abstrahieren Merkmale
- **Ausgabeschicht:** Gibt die finale Vorhersage aus (z. B. Kategorie, Wert)

Netze mit mehreren versteckten Schichten werden als **Mehrschicht-Perzeptrons (MLP)** bezeichnet. Sie gehören zur Klasse der **Feedforward-Netze**, bei denen Informationen nur in eine Richtung fließen – von der Eingabe zur Ausgabe.

![image](https://github.com/user-attachments/assets/8d540b6c-2be0-49f7-bbd7-b2d951789290)

*Feedforward-Netz mit 2 Hidden Layers – Quelle: ORDIX Blog, 2021*


**Erklärung:**  

Die Eingabeschicht (blau) nimmt die Merkmale x1 bis x5 auf. Diese werden durch zehn Neuronen in der ersten verdeckten Schicht verarbeitet, anschließend durch fünf weitere Neuronen in der zweiten versteckten Schicht weitergegeben und schließlich zu einer Vorhersage y1 in der Ausgabeschicht (grün) zusammengeführt. Jede Verbindung ist mit einem Gewicht versehen, das beim Lernen angepasst wird.

---

## Anwendung: Bilderkennung

Ein neuronales Netz kann z. B. lernen, **handgeschriebene Ziffern (0–9)** zu erkennen:

1. **Eingabeschicht:** nimmt Pixelwerte des Bildes auf  
2. **Versteckte Schichten:** extrahieren Muster wie Kanten oder Kurven  
3. **Ausgabeschicht:** entscheidet sich für die wahrscheinlichste Ziffer

---

> Ein neuronales Netz lernt durch viele Beispiele, relevante Merkmale in den Daten zu erkennen – ganz ohne explizit programmierte Regeln.



## 6.2 Lernen mit Backpropagation

Damit ein neuronales Netz nicht nur zufällige Ausgaben produziert, sondern gezielt Muster erkennen kann, muss es aus Beispielen lernen. Das zentrale Trainingsverfahren für Feedforward-Netze ist der **Backpropagation-Algorithmus** – auf Deutsch: Rückwärtsausbreitung des Fehlers.

### Ziel: Fehler minimieren

Der Lernprozess basiert auf dem Vergleich zwischen der tatsächlichen Ausgabe \( \hat{y} \) des Netzes und dem Zielwert \( y \). Die Differenz wird durch eine **Fehlerfunktion (Loss Function)** quantifiziert, z. B.:

![image](https://github.com/user-attachments/assets/71d425bc-95e2-435b-ad2a-7f666f5ab9ad)


Ziel ist es, die Summe der Fehler über alle Trainingsbeispiele hinweg zu minimieren. Dazu passt das Netz seine **Gewichte** iterativ an.

---

### Idee der Rückwärtsausbreitung

Backpropagation nutzt die **Kettenregel** der Differentialrechnung, um zu berechnen, wie stark jedes Gewicht zum Fehler beigetragen hat. Der Fehler wird dabei von der **Ausgabe zurück zur Eingabe** propagiert.
Jedes Gewicht \( w \) erhält ein individuelles **Gradientenmaß**, das angibt, in welche Richtung es angepasst werden sollte.

Der Anpassungsschritt erfolgt dann mit Hilfe des Gradientenabstiegs:

![image](https://github.com/user-attachments/assets/9eedce74-ca79-4b24-8250-191afc1636d0)

Dabei ist η die Lernrate, die steuert, wie stark die Gewichte verändert werden. Ist sie zu groß, „springt“ das Netz über das Minimum hinaus; ist sie zu klein, dauert das Lernen sehr lange.

![image](https://github.com/user-attachments/assets/7c2fa529-b1e3-4d97-85d4-182e51366c2a)

*Abbildung: Backpropagation in einem neuronalen Netz – Quelle: GeeksforGeeks (2023)*

---

### Beispiel: Spam-Klassifikation mit einem MLP

Ein **Mehrschicht-Perzeptron (MLP)** soll erkennen, ob eine E-Mail Spam ist. In der Trainingsphase erhält das Netz hunderte Beispiele mit Labels:

- „Spam“  
- „Kein Spam“

Der Ablauf:

1. Vorhersage durch das Netz  
2. Vergleich mit dem Label  
3. Fehlerberechnung  
4. Fehler wird durch das Netz zurückgeleitet  
5. Gewichte werden aktualisiert

Mit jeder Iteration verbessert sich die Erkennung. Das Netz lernt, welche Merkmalskombinationen typisch für Spam sind – z. B. bestimmte Wörter im Betreff.

---

### Bedeutung für moderne Netze

**Backpropagation** ist das Herzstück nahezu aller modernen neuronalen Netze:

- **MLPs**
- **Convolutional Neural Networks (CNNs)**
- **Recurrent Neural Networks (RNNs)**
- u. v. m.

Ohne diesen Algorithmus wäre das effiziente Training tiefer Modelle kaum möglich.

### Erweiterte Optimierungsstrategien

Zur Verbesserung des Lernverhaltens kommen häufig zusätzliche Methoden zum Einsatz:

- **Momentum:** beschleunigt den Lernprozess  
- **Adam (Adaptive Moment Estimation):** kombiniert Momentum und RMSprop  
- **Early Stopping:** verhindert Überanpassung durch Abbruch bei stagnierender Validierungsleistung

> Backpropagation + geeignete Optimierungsstrategien = Grundlage moderner Deep Learning Systeme


## 6.3 Tiefe neuronale Netze (Deep Neural Networks)

Tiefe neuronale Netze – häufig als **Deep Neural Networks (DNNs)** bezeichnet – bilden das Herzstück moderner Deep-Learning-Anwendungen. Sie erweitern klassische künstliche neuronale Netze (ANNs) um eine **deutlich größere Anzahl an versteckten Schichten (Hidden Layers)**.

Während einfache Feedforward-Netze meist nur eine oder zwei versteckte Schichten besitzen, verwenden DNNs oft **dutzende Layer**, um zunehmend komplexere Merkmale zu lernen.

---

### Merkmale tiefer Netze

- **Hierarchisches Lernen:**  
  Frühere Schichten lernen einfache Merkmale (z. B. Kanten, Farben). Spätere Schichten kombinieren diese zu komplexeren Repräsentationen (z. B. Gesichter, Objekte, Sprache).

- **Nichtlinearität durch Aktivierungsfunktionen:**  
  Häufig eingesetzt: **ReLU (Rectified Linear Unit)**  
  Vorteile:
  - Vermeidet das Problem des **verschwindenden Gradienten**
  - Führt zu schnellerer Konvergenz beim Lernen

- **Hoher Bedarf an Rechenleistung und Daten:**  
  - Erfordert **große Mengen gelabelter Daten** (z. B. Millionen Bilder)
  - Training erfolgt auf leistungsstarker Hardware: **GPU** oder **TPU**

---

### Vergleich: Konventionelles ML vs. Deep Learning vs. LLMs

| Merkmal                  | Traditionelles ML | Deep Learning (DNNs) | LLMs (z. B. ChatGPT) |
|--------------------------|-------------------|------------------------|-----------------------|
| Trainingsdatenmenge      | Groß              | Groß                   | Sehr groß             |
| Feature Engineering      | Manuell           | Automatisch            | Automatisch           |
| Modellkomplexität        | Begrenzt          | Hoch                   | Extrem hoch           |
| Interpretierbarkeit      | Gut               | Schwach                | Sehr schwach          |
| Leistung                 | Mittelmäßig       | Hoch                   | Sehr hoch             |
| Hardwareanforderungen    | Gering            | Hoch                   | Sehr hoch             |

*Quelle: 430_DL_DNN, Seite 24*

---

### Anwendungen tiefer Netze

DNNs kommen in vielen Bereichen zum Einsatz:

- 🗣 **Sprachverarbeitung**  
  *Beispiele:* Google Translate, ChatGPT, Alexa

- 🩺 **Medizinische Bildanalyse**  
  *Beispiel:* Erkennung von Tumoren oder Anomalien in Röntgenbildern

- 💳 **Finanzwesen**  
  *Beispiel:* Betrugserkennung bei Kreditkartentransaktionen

- 🚗 **Autonomes Fahren**  
  *Beispiel:* Bilderkennung, Objekterkennung, Sensorfusion

- 📷 **Bild- und Objekterkennung**  
  *Beispiel:* Klassifikation in der Industrie, Sicherheitsanwendungen

---

> Tiefe neuronale Netze sind heute aus der KI nicht mehr wegzudenken – sie liefern die Grundlage für viele der fortschrittlichsten Technologien unserer Zeit.

## 6.4 Convolutional Neural Networks (CNNs)

**Convolutional Neural Networks (CNNs)** sind speziell für die Verarbeitung visueller Daten (z. B. Bilder, Videos) entwickelte tiefe neuronale Netze. Ihre Architektur ermöglicht es, **lokale visuelle Merkmale** automatisch zu erkennen und schrittweise zu abstrahieren – ideal für Klassifikations-, Erkennungs- und Segmentierungsaufgaben.

Im Gegensatz zu vollständig verbundenen Netzen nutzen CNNs **Faltungsschichten (Convolutional Layers)** und **Pooling-Schichten**, wodurch die Anzahl der Parameter und die Komplexität reduziert wird.

---

### Architektur und Datenfluss

![image](https://github.com/user-attachments/assets/5c88fbfa-d8b1-4841-8480-c0872d347367)
 
*Abbildung: CNN-Aufbau von der Eingabe bis zur Klassifikation – Quelle: MathWorks*

**Ablauf eines CNNs:**

1. **Convolutional Layer:**  
   Extrahiert lokale Merkmale (z. B. Kanten, Texturen) mithilfe von Filtern

2. **ReLU-Aktivierung:**  
   Führt Nichtlinearität ein und hilft beim Training tiefere Strukturen zu lernen

3. **Pooling Layer:**  
   Reduziert die Dimensionalität (z. B. mit MaxPooling) und erhöht die Translationstoleranz

4. **Wiederholung von Convolution + Pooling:**  
   Merkmale werden hierarchisch komplexer – von Linien zu ganzen Objekten

5. **Flatten Layer:**  
   Wandelt Merkmalskarten in einen Vektor um

6. **Dense (Fully Connected) Layer:**  
   Verknüpft alle Neuronen zur Entscheidungsbildung

7. **Softmax-Ausgabe:**  
   Gibt Wahrscheinlichkeiten für jede Klasse aus (z. B. „Auto“, „LKW“, „Fahrrad“)

---

### Feature Learning & Klassifikation

Die CNN-Architektur lässt sich in zwei Hauptphasen gliedern:

- **Feature Learning:**  
  Merkmale werden automatisch erkannt – vom Pixel über Kanten bis zu komplexen Objekten

- **Classification:**  
  In der vollständig verbundenen Endphase wird entschieden, zu welcher Klasse das Bild gehört – meist durch **Softmax**.

---

### Vorteile von CNNs

✅ **Lokalität:** Durch Filter beschränkt auf kleine Bildausschnitte  
✅ **Parameterersparnis:** Gewichtsteilung reduziert Rechenaufwand  
✅ **Translationstoleranz:** Pooling macht das Netz robust gegenüber Objektverschiebung  
✅ **Hohe Genauigkeit:** Besonders bei Bildklassifikation & Objekterkennung

---

### Typische Einsatzgebiete

- 📸 Bildklassifikation (z. B. ImageNet)
- 🔍 Objekterkennung (z. B. YOLO, Faster R-CNN)
- 🧠 Medizinische Bilddiagnostik (z. B. Tumorerkennung)
- 🚗 Autonomes Fahren (z. B. Verkehrsschilderkennung)
- 📹 Videoanalyse & Gesichtserkennung

---

## Quellen

1. ORDIX Blog (2021): Einstieg in neuronale Netze mit TensorFlow und Keras  
   [https://blog.ordix.de/einstieg-in-neuronale-netze-mit-tensorflow-und-keras](https://blog.ordix.de/einstieg-in-neuronale-netze-mit-tensorflow-und-keras)

2. Rosenblatt, F. (1958): *The Perceptron*  
   Psychological Review, Vol. 65, No. 6

3. Goodfellow, I., Bengio, Y., Courville, A. (2016): *Deep Learning*  
   [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

4. Mitchell, T. (1997): *Machine Learning*, McGraw-Hill Education

5. Kriegel, H.-P. et al. (2020): *Künstliche Intelligenz: Grundlagen intelligenter Systeme*, Springer Vieweg

6. Wikipedia (2025): [Künstliches neuronales Netz](https://de.wikipedia.org/wiki/Künstliches_neuronales_Netz)

7. GeeksforGeeks (2023): [Backpropagation in Neural Network](https://www.geeksforgeeks.org/backpropagation-in-neural-network/)

