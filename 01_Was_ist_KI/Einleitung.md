# 1. Einführung in die Künstliche Intelligenz 🤖

Spätestens seit der Veröffentlichung von **ChatGPT im Jahr 2022** ist der Begriff *Künstliche Intelligenz (KI)* stark in den Fokus gerückt. In Medien, Unternehmen und Bildungseinrichtungen wird KI heute als Schlüsseltechnologie diskutiert, die Wirtschaft, Gesundheitswesen und Alltag nachhaltig verändern kann.

Dabei ist die Idee nicht neu: Bereits in den **1950er-Jahren** beschäftigten sich Wissenschaftler mit der Frage, ob Maschinen intelligentes Verhalten zeigen könnten. Die technologische Grundlage hat sich seither stetig weiterentwickelt – von symbolischen Systemen über
maschinelles Lernen bis hin zu heutigen neuronalen Netzen. Doch was genau ist
Künstliche Intelligenz?

---

## 1.1 Was ist Künstliche Intelligenz?

**Künstliche Intelligenz** bezeichnet den Versuch, Maschinen mit Fähigkeiten auszustatten, die typischerweise dem Menschen vorbehalten sind – etwa:

* das **Verstehen natürlicher Sprache**,
* das **Lernen aus Erfahrung**,
* **Problemlösen**,
* **Planen und Entscheiden**,
* **Wahrnehmen und Handeln**.

### Definitionen:

> 🧠 **John McCarthy** (1956):
> "Die Wissenschaft und Technik, intelligente Maschinen zu bauen, insbesondere intelligente Computerprogramme."

> 📘 **Patrick Winston** (1992):
> "KI ist die Untersuchung von Berechnungsverfahren, die es ermöglichen, wahrzunehmen, zu denken und zu handeln."

> 🌐 **Wikipedia (2025)**:
> "KI ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens und dem maschinellen Lernen befasst."

### Merkmale:

KI ist kein einheitlicher Algorithmus, sondern ein **Sammelbegriff** für verschiedene Methoden:

* Symbolische KI (z. B. logikbasierte Systeme),
* Statistikbasierte Verfahren (z. B. Entscheidungsbäume),
* Maschinelles Lernen und Deep Learning.

Was zu einem bestimmten Zeitpunkt als "intelligent" gilt, **verändert sich mit dem technischen Fortschritt**. Viele Systeme, die früher als KI galten (z. B. Taschenrechner oder Schachprogramme), werden heute als normale Software angesehen.

---

## 1.2 Arten von KI

Künstliche Intelligenz lässt sich auf mehreren Ebenen klassifizieren – je nach Zielsetzung, Komplexität und Flexibilität des Systems. Eine gängige Unterscheidung erfolgt entlang des Spektrums von „schwacher“ bis „starker“ KI. Technisch differenziert man insbesondere zwischen drei Haupttypen:

**1. Artificial Narrow Intelligence (ANI)**

Die heute existierenden KI-Systeme gehören fast ausschließlich zur Kategorie der schwachen oder spezialisierten KI (ANI). Sie sind darauf ausgelegt, konkrete Aufgaben in eng definierten Anwendungsfeldern zu erfüllen – z. B. Spracherkennung, Bilderkennung, maschinelle Übersetzung oder Produktempfehlungen. Trotz ihrer Leistungsfähigkeit agieren ANI-Systeme immer innerhalb vorgegebener Grenzen und verfügen über kein „Verständnis“ im menschlichen Sinn. Klassische Beispiele sind
Spamfilter, Chatbots oder Systeme zur medizinischen Bildanalyse.

**2. Artificial General Intelligence (AGI)**

Die sogenannte starke KI bezeichnet eine hypothetische Form künstlicher Intelligenz, die in der Lage wäre, flexibel in verschiedensten Kontexten zu denken, zu lernen und zu handeln – ähnlich einem menschlichen Geist. Eine AGI könnte Wissen aus einem Bereich auf völlig andere übertragen, kreativ neue Lösungen entwickeln und eigene Ziele verfolgen. Der Stand der Forschung ist hier rein experimentell: Es gibt bisher kein
System, das als AGI anerkannt werden kann.

**3. Artificial Super Intelligence (ASI)**

Ein weiterer theoretischer Schritt über AGI hinaus ist die Superintelligenz – eine Form von KI, die den Menschen in nahezu allen kognitiven Bereichen übertreffen würde. In der Literatur wird sie oft mit Chancen (z. B. wissenschaftliche Durchbrüche) und Risiken (z. B. Kontrollverlust) diskutiert. Technisch ist ASI derzeit rein spekulativ und Gegenstand ethisch-philosophischer Debatten, nicht aber praktischer Entwicklung.



| Typ                                         | Beschreibung                                                                      | Stand           |
| ------------------------------------------- | --------------------------------------------------------------------------------- | --------------- |
| **ANI** (*Artificial Narrow Intelligence*)  | Spezialisiert auf eine Aufgabe (z. B. Spracherkennung, Bilderkennung, Chatbots)   | ✅ Realität      |
| **AGI** (*Artificial General Intelligence*) | Allgemeine Intelligenz wie beim Menschen, kann flexibel in neuen Kontexten lernen | 🔬 Hypothetisch |
| **ASI** (*Artificial Super Intelligence*)   | Übermenschliche Intelligenz, überlegen in allen kognitiven Bereichen              | ⚠️ Spekulativ   |

---

## 1.3 GPT-Architektur

Moderne Sprachmodelle wie **ChatGPT** basieren auf der sogenannten **Transformer-Architektur**, die 2017 von Vaswani et al. eingeführt wurde ([„Attention is All You Need“](https://arxiv.org/abs/1706.03762)).

GPT steht für „Generative Pretrained Transformer“. Das Modell lernt, Sprachmuster vorherzusagen, indem es riesige Mengen an Textdaten analysiert. Dabei wird keine Bedeutung „verstanden“, sondern **Wahrscheinlichkeiten für das nächste Wort** vorhergesagt.

 ![image](https://github.com/user-attachments/assets/d496fc24-a5a9-4368-860c-437ed9016358)

*Abbildung 1: Die GPT-Architektur basiert auf einem Transformer-Modell. Quelle: Jalammar (2020)*


---

## 1.4 Der Turing-Test

🧪 **Alan Turing** stellte 1950 die berühmte Frage: *„Can machines think?“*. Zur Beantwortung schlug er den **Turing-Test** vor:

Ein Mensch kommuniziert über ein Terminal mit zwei Gesprächspartnern – einem Menschen und einer Maschine. Kann der Mensch nicht zuverlässig sagen, wer die Maschine ist, gilt der Test als bestanden.

![image](https://github.com/user-attachments/assets/91854e0b-a4f1-48e0-97a0-88f79c0a1475)

*Abbildung 2: Illustration des Turing-Tests. Quelle: GeeksforGeeks (2023)*


### Ist der Test noch zeitgemäß?

Zwar war der Turing-Test ein **bahnbrechendes Konzept**, doch moderne Modelle wie ChatGPT bestehen diesen Test **ohne echtes Verständnis**. Sie imitieren Sprache, ohne Bewusstsein oder logische Tiefe. Daher gilt heute:

> *"Der Turing-Test ist ein erster Maßstab – aber nicht mehr ausreichend für moderne KI."*

---

## 🔍 Quellen

* McCarthy, J. (1956): *Proposal for the Dartmouth Summer Research Project on AI*
* Winston, P. (1992): *Artificial Intelligence*, Addison-Wesley
* Vaswani et al. (2017): *Attention is All You Need*, [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
* Wikipedia (2025): [https://de.wikipedia.org/wiki/Künstliche\_Intelligenz](https://de.wikipedia.org/wiki/Künstliche_Intelligenz)
* GeeksforGeeks (2023): *Turing Test in AI*, [https://www.geeksforgeeks.org/turing-test-artificial-intelligence/](https://www.geeksforgeeks.org/turing-test-artificial-intelligence/)

