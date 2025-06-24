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



## Netzstruktur: Von der Eingabe zur Vorhersage

Ein einzelnes Neuron ist nicht ausreichend für komplexe Aufgaben. Erst durch das **Verknüpfen mehrerer Neuronen zu Netzwerken** entstehen leistungsfähige Strukturen:

### Typische Architektur eines neuronalen Netzes:

- **Eingabeschicht:** Nimmt Rohdaten auf (z. B. Pixel, Sensordaten)
- **Versteckte Schichten (Hidden Layers):** Transformieren und abstrahieren Merkmale
- **Ausgabeschicht:** Gibt die finale Vorhersage aus (z. B. Kategorie, Wert)

Netze mit mehreren versteckten Schichten werden als **Mehrschicht-Perzeptrons (MLP)** bezeichnet. Sie gehören zur Klasse der **Feedforward-Netze**, bei denen Informationen nur in eine Richtung fließen – von der Eingabe zur Ausgabe.

![image](https://github.com/user-attachments/assets/8d540b6c-2be0-49f7-bbd7-b2d951789290)

*Abbildung 11: Feedforward-Netz mit 2 Hidden Layers – Quelle: ORDIX Blog, 2021*


**Erklärung:**  

Die Eingabeschicht (blau) nimmt die Merkmale x1 bis x5 auf. Diese werden durch zehn Neuronen in der ersten verdeckten Schicht verarbeitet, anschließend durch fünf weitere Neuronen in der zweiten versteckten Schicht weitergegeben und schließlich zu einer Vorhersage y1 in der Ausgabeschicht (grün) zusammengeführt. Jede Verbindung ist mit einem Gewicht versehen, das beim Lernen angepasst wird.




## Anwendung: Bilderkennung

Ein neuronales Netz kann z. B. lernen, **handgeschriebene Ziffern (0–9)** zu erkennen:

1. **Eingabeschicht:** nimmt Pixelwerte des Bildes auf  
2. **Versteckte Schichten:** extrahieren Muster wie Kanten oder Kurven  
3. **Ausgabeschicht:** entscheidet sich für die wahrscheinlichste Ziffer



> Ein neuronales Netz lernt durch viele Beispiele, relevante Merkmale in den Daten zu erkennen – ganz ohne explizit programmierte Regeln.

---

## 6.2 Lernen mit Backpropagation

Damit ein neuronales Netz nicht nur zufällige Ausgaben produziert, sondern gezielt Muster erkennen kann, muss es aus Beispielen lernen. Das zentrale Trainingsverfahren für Feedforward-Netze ist der **Backpropagation-Algorithmus** – auf Deutsch: Rückwärtsausbreitung des Fehlers.

### Ziel: Fehler minimieren

Der Lernprozess basiert auf dem Vergleich zwischen der tatsächlichen Ausgabe \( \hat{y} \) des Netzes und dem Zielwert \( y \). Die Differenz wird durch eine **Fehlerfunktion (Loss Function)** quantifiziert, z. B.:

![image](https://github.com/user-attachments/assets/71d425bc-95e2-435b-ad2a-7f666f5ab9ad)


Ziel ist es, die Summe der Fehler über alle Trainingsbeispiele hinweg zu minimieren. Dazu passt das Netz seine **Gewichte** iterativ an.



### Idee der Rückwärtsausbreitung

Backpropagation nutzt die **Kettenregel** der Differentialrechnung, um zu berechnen, wie stark jedes Gewicht zum Fehler beigetragen hat. Der Fehler wird dabei von der **Ausgabe zurück zur Eingabe** propagiert.
Jedes Gewicht \( w \) erhält ein individuelles **Gradientenmaß**, das angibt, in welche Richtung es angepasst werden sollte.

Der Anpassungsschritt erfolgt dann mit Hilfe des Gradientenabstiegs:

![image](https://github.com/user-attachments/assets/9eedce74-ca79-4b24-8250-191afc1636d0)

Dabei ist η die Lernrate, die steuert, wie stark die Gewichte verändert werden. Ist sie zu groß, „springt“ das Netz über das Minimum hinaus; ist sie zu klein, dauert das Lernen sehr lange.

![image](https://github.com/user-attachments/assets/7c2fa529-b1e3-4d97-85d4-182e51366c2a)

*Abbildung 12: Backpropagation in einem neuronalen Netz – Quelle: GeeksforGeeks (2023)*



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

---

## 6.3 Tiefe neuronale Netze (Deep Neural Networks)

Tiefe neuronale Netze – häufig als **Deep Neural Networks (DNNs)** bezeichnet – bilden das Herzstück moderner Deep-Learning-Anwendungen. Sie erweitern klassische künstliche neuronale Netze (ANNs) um eine **deutlich größere Anzahl an versteckten Schichten (Hidden Layers)**.

Während einfache Feedforward-Netze meist nur eine oder zwei versteckte Schichten besitzen, verwenden DNNs oft **dutzende Layer**, um zunehmend komplexere Merkmale zu lernen.



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



> Tiefe neuronale Netze sind heute aus der KI nicht mehr wegzudenken – sie liefern die Grundlage für viele der fortschrittlichsten Technologien unserer Zeit.

---

## 6.4 Convolutional Neural Networks (CNNs)

**Convolutional Neural Networks (CNNs)** sind speziell für die Verarbeitung visueller Daten (z. B. Bilder, Videos) entwickelte tiefe neuronale Netze. Ihre Architektur ermöglicht es, **lokale visuelle Merkmale** automatisch zu erkennen und schrittweise zu abstrahieren – ideal für Klassifikations-, Erkennungs- und Segmentierungsaufgaben.

Im Gegensatz zu vollständig verbundenen Netzen nutzen CNNs **Faltungsschichten (Convolutional Layers)** und **Pooling-Schichten**, wodurch die Anzahl der Parameter und die Komplexität reduziert wird.



### Architektur und Datenfluss

![image](https://github.com/user-attachments/assets/5c88fbfa-d8b1-4841-8480-c0872d347367)
 
*Abbildung 13: CNN-Aufbau von der Eingabe bis zur Klassifikation – Quelle: MathWorks*

**Ablauf eines CNNs:**

Ein Convolutional Neural Network (CNN) verarbeitet Bilddaten in mehreren aufeinanderfolgenden Schritten. Zunächst wird das Eingabebild durch Faltungsschichten geleitet, in denen Filter lokale Merkmale wie Kanten oder Formen erkennen. Anschließend reduziert eine Pooling-Schicht die Bildgröße, wobei wichtige Informationen erhalten bleiben. Dieser Prozess – bestehend aus Faltung, Aktivierungsfunktion (ReLU) und Pooling – wird mehrfach wiederholt, um zunehmend komplexere Merkmale zu extrahieren.

Die daraus entstehenden Merkmalskarten werden anschließend durch eine Flatten-Schicht in einen Vektor überführt und an vollständig verbundene Neuronen (Dense Layer) übergeben. Diese führen auf Basis der extrahierten Merkmale die Klassifikation durch. Am Ende sorgt eine Softmax-Schicht dafür, dass das Netzwerk eine Wahrscheinlichkeitsverteilung über alle möglichen Klassen (z. B. Auto, LKW, Fahrrad) ausgibt.



### Feature Learning & Klassifikation

Die CNN-Architektur lässt sich in zwei Hauptphasen gliedern:

- **Feature Learning:**  
  Merkmale werden automatisch erkannt – vom Pixel über Kanten bis zu komplexen Objekten

- **Classification:**  
  In der vollständig verbundenen Endphase wird entschieden, zu welcher Klasse das Bild gehört – meist durch **Softmax**.



### Vorteile von CNNs

✅ **Lokalität:** Durch Filter beschränkt auf kleine Bildausschnitte  
✅ **Parameterersparnis:** Gewichtsteilung reduziert Rechenaufwand  
✅ **Translationstoleranz:** Pooling macht das Netz robust gegenüber Objektverschiebung  
✅ **Hohe Genauigkeit:** Besonders bei Bildklassifikation & Objekterkennung



### Typische Einsatzgebiete

- 📸 Bildklassifikation (z. B. ImageNet)
- 🔍 Objekterkennung (z. B. YOLO, Faster R-CNN)
- 🧠 Medizinische Bilddiagnostik (z. B. Tumorerkennung)
- 🚗 Autonomes Fahren (z. B. Verkehrsschilderkennung)
- 📹 Videoanalyse & Gesichtserkennung

---

## 6.5 Natürliche Sprachverarbeitung (NLP)

Die natürliche Sprachverarbeitung (Natural Language Processing, kurz: NLP) ist ein zentrales Anwendungsfeld der künstlichen Intelligenz, das sich mit der automatisierten Analyse, Interpretation und Generierung von menschlicher Sprache beschäftigt. Ziel von NLP-Systemen ist es, Texte oder gesprochene Sprache so zu verarbeiten, dass Computer deren Inhalte „verstehen“ und sinnvoll darauf reagieren können.

### Grundlagen und Anwendungsbereiche

NLP basiert auf linguistischen Regeln sowie statistischen und maschinellen Lernverfahren. Dabei werden Texte zunächst in ihre Bestandteile zerlegt und schrittweise verarbeitet. Typische Aufgaben der natürlichen Sprachverarbeitung sind:

- **Tokenisierung:** Zerlegung eines Textes in einzelne Wörter oder Sätze.  
- **Lemmatisierung und Stemming:** Reduktion von Wörtern auf ihre Grundform (z. B. „lief“ → „laufen“).  
- **Part-of-Speech-Tagging:** Bestimmung der Wortarten (z. B. Substantiv, Verb).  
- **Parsing:** Analyse der grammatikalischen Struktur eines Satzes.  
- **Named Entity Recognition (NER):** Erkennung von Eigennamen, Orten oder Zeitangaben.

Diese Schritte bilden die Grundlage für komplexere Anwendungen wie automatische Übersetzung, Chatbots, Sentiment-Analyse oder Textzusammenfassungen.

### Sprachmodelle und moderne Architektur: Transformer

Moderne NLP-Systeme basieren überwiegend auf neuronalen Netzen, insbesondere Transformer-Architekturen. Diese gelten seit 2017 als Standard und bilden die Grundlage für viele große Sprachmodelle wie BERT, GPT oder T5.

Die folgende Abbildung zeigt den schematischen Aufbau eines klassischen Transformers:

<img width="267" alt="GENAI-1 151ded5440b4c997bac0642ec669a00acff2cca1" src="https://github.com/user-attachments/assets/219a5ff8-d976-4765-b962-f68cf7c80153" />

*Abbildung 14: Aufbau eines Transformer-Modells mit Encoder und Decoder -  Quelle: Amazon Web Services (2024): „Was sind Transformer in der künstlichen Intelligenz?“*

Der Transformer besteht aus zwei Hauptkomponenten:

- **Encoder (links):** Wandelt die Eingabesequenz (Input Embedding + Positionskodierung) durch mehrschichtige Verarbeitung in eine abstrakte Repräsentation um.  
- **Decoder (rechts):** Generiert auf Basis dieser Repräsentation ein Ausgabeergebnis (z. B. Übersetzung).

Die wichtigsten Elemente im Modell sind:

- **Multi-Head Attention:** Der Kernmechanismus, um Kontextbeziehungen zwischen Wörtern zu erkennen – auch über große Distanzen im Satz.  
- **Masked Multi-Head Attention:** Im Decoder sorgt diese Variante dafür, dass bei der Generierung eines Satzes nur bereits erzeugte Wörter berücksichtigt werden (Kausalität).  
- **Feed Forward Layer:** Zuständig für nicht-lineare Transformationen innerhalb jeder Ebene.  
- **Add & Norm:** Normalisierungsschritte zur Stabilisierung des Trainings.  
- **Positional Encoding:** Da das Modell selbst keine Reihenfolge kennt, wird die Wortposition explizit codiert.

### Bedeutung in der Praxis

NLP ist heute in zahlreichen digitalen Anwendungen allgegenwärtig – von Rechtschreibkorrekturen über automatische Übersetzungen (z. B. DeepL, Google Translate) bis hin zu Sprachassistenten wie Siri oder Alexa. Auch in der Finanzbranche, im Gesundheitswesen und in der juristischen Dokumentenanalyse gewinnt NLP zunehmend an Bedeutung.

---

## 6.6 Große Sprachmodelle und Embeddings (LLMs)

In den letzten Jahren haben sich sogenannte große Sprachmodelle (engl. „Large Language Models“ – kurz: LLMs) zu einem zentralen Bestandteil moderner KI-Systeme entwickelt. Diese Modelle sind in der Lage, menschliche Sprache auf beeindruckende Weise zu verstehen und sogar eigenständig Texte zu generieren. Grundlage dafür ist die Transformer-Architektur, auf die bereits im vorherigen Kapitel eingegangen wurde.

### Was genau sind LLMs?

Große Sprachmodelle bestehen aus neuronalen Netzwerken mit mehreren Milliarden Parametern. Trainiert werden sie auf riesigen Mengen an Text – darunter Bücher, Webseiten, Forenbeiträge oder Wikipedia-Einträge. Während des Trainings lernen sie, Wahrscheinlichkeiten für Wortfolgen vorherzusagen. Das klingt zunächst unspektakulär, führt aber dazu, dass sie in der Lage sind, Sprache erstaunlich gut nachzubilden – oft so überzeugend, dass man meinen könnte, ein Mensch habe den Text geschrieben.

Bekannte Beispiele für solche Modelle sind **GPT** (von OpenAI), **BERT** (von Google) oder **LLaMA** (von Meta). Viele dieser Modelle sind öffentlich zugänglich oder in Plattformen wie Chatbots oder Suchmaschinen integriert.

### Die Rolle von Embeddings

Damit ein Sprachmodell mit Wörtern überhaupt arbeiten kann, müssen diese zunächst mathematisch dargestellt werden – und zwar in Form sogenannter **Embeddings**. Dabei wird jedes Wort (oder auch ein Wortbestandteil) als ein mehrdimensionaler Vektor dargestellt. Der Clou: Wörter mit ähnlicher Bedeutung liegen in diesem Vektorraum näher beieinander als solche mit unterschiedlicher Bedeutung.

Frühere Verfahren wie **Word2Vec** oder **GloVe** haben für jedes Wort einen festen Vektor erzeugt – unabhängig vom Kontext. Moderne Modelle wie **BERT** oder **GPT** berücksichtigen dagegen den jeweiligen Satzzusammenhang. So bekommt das Wort *„Bank“* in einem Satz über Geld eine andere Repräsentation als in einem Satz über einen Park.

### Aufbau und Lernprozess

Ein LLM besteht aus vielen übereinandergeschichteten **Transformer-Blöcken**. Jeder dieser Blöcke verarbeitet die Eingabedaten und gibt sie an die nächste Ebene weiter. Dabei werden die Beziehungen zwischen Wörtern analysiert und gewichtet – ein Mechanismus, der als **Self-Attention** bekannt ist.

Der Lernprozess ist zweigeteilt:

- **Pretraining**: Das Modell wird allgemein trainiert, um Sprache, Grammatik und Weltwissen zu lernen.  
- **Finetuning**: Danach kann es gezielt auf bestimmte Aufgaben vorbereitet werden, z. B. für den Einsatz in der Medizin, im Recht oder im Kundenservice.

### Wichtig zu wissen

Diese Systeme arbeiten nicht „intelligent“ im menschlichen Sinne. Sie haben **kein eigenes Verständnis**, sondern verarbeiten Sprache rein statistisch. Trotzdem sind die Ergebnisse oft so gut, dass sie in vielen Bereichen einen echten Mehrwert bieten.

---

## 6.7 Prompt Engineering und Anwendungsmöglichkeiten

Ein Aspekt im Umgang mit großen Sprachmodellen (LLMs) ist die sogenannte **„Prompt-Steuerung“**. Dabei geht es darum, wie man einem Modell Anweisungen gibt – also wie man die Eingabeformulierung (den Prompt) gestaltet, um möglichst sinnvolle, präzise oder kreative Antworten zu erhalten. Dieses Vorgehen wird unter dem Begriff **Prompt Engineering** zusammengefasst.

### Warum ist Prompt Engineering wichtig?

Sprachmodelle reagieren sehr sensibel auf die Art und Weise, wie Fragen gestellt oder Aufgaben beschrieben werden. Schon kleine Änderungen in der Formulierung können zu völlig unterschiedlichen Ergebnissen führen. Wer gute Resultate möchte, muss daher lernen, mit dem Modell klar, zielgerichtet und strategisch zu kommunizieren.

Ein einfaches Beispiel:

- „Erkläre mir kurz, was ein neuronales Netz ist.“  
- „Stell dir vor, du erklärst einem Schüler in der 9. Klasse, was ein neuronales Netz macht. Bitte möglichst anschaulich.“

Beide Prompts haben das gleiche Ziel, doch die zweite Variante liefert oft die verständlichere Antwort – weil sie dem Modell mehr Kontext und eine klare Rolle vorgibt.

### Techniken und Strategien

Inzwischen gibt es eine Reihe von erprobten Methoden, um Prompts gezielt zu gestalten. Hier ein paar wichtige Beispiele:

- **Few-Shot Learning:** Man zeigt dem Modell vorab ein paar Beispiele (z. B. Fragen und Antworten), bevor man die eigentliche Aufgabe stellt.  
- **Chain-of-Thought Prompting:** Das Modell wird aufgefordert, einen Denkprozess in mehreren Schritten durchzuführen, bevor es ein Ergebnis nennt. Besonders hilfreich bei komplexen Aufgaben oder Rechenwegen.  
- **Role-Based Prompting:** Dem Modell wird eine bestimmte Rolle zugewiesen (z. B. „Du bist ein IT-Experte“ oder „Du schreibst als Lehrer“), um Stil und Inhalt zu beeinflussen.  
- **Prompt Chaining:** Komplexe Aufgaben werden in kleinere Teilschritte zerlegt, die nacheinander abgearbeitet werden. Das verbessert oft die Qualität und Nachvollziehbarkeit.

### Vorischt

- Sprachmodelle neigen gelegentlich zu sogenannten **Halluzinationen** – also erfundenen oder sachlich falschen Informationen, die aber sprachlich überzeugend klingen.  
- Bei schlecht formulierten Prompts kann es zu **Fehlinterpretationen oder nicht reproduzierbaren Ergebnissen** kommen.  
- Zudem besteht die Gefahr sogenannter **Prompt Injections**, bei denen ein Modell gezielt durch manipulierte Eingaben zu unerwünschtem Verhalten gebracht wird.

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

8. Amazon Web Services (2024): *Was sind Transformer in der künstlichen Intelligenz? (Abbildung 16)*  
   https://aws.amazon.com/de/what-is/transformers-in-artificial-intelligence/

9. Vaswani, A. et al. (2017): *Attention is All You Need.*  
   https://arxiv.org/abs/1706.03762

10. OpenAI (2023): *Best Practices for Prompt Engineering with GPT.*  
    https://platform.openai.com/docs/guides/gpt-best-practices
11. MathWorks (o. J.): *Convolutional Neural Networks – Aufbau eines CNNs von der Eingabe bis zur Klassifikation * https://de.mathworks.com/discovery/convolutional-neural-network.html
 
