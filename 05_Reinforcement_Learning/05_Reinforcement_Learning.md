# 5. Bestärkendes Lernen – Reinforcement Learning

## 5.1 Grundprinzip

Beim **bestärkenden Lernen** (Reinforcement Learning, RL) lernt ein Agent durch Interaktion mit seiner Umgebung. Ziel ist es, eine **Strategie (Policy)** zu entwickeln, die langfristig möglichst viel Belohnung maximiert.

Nach jeder Aktion erhält der Agent eine Rückmeldung (Reward) in Form von Belohnung oder Bestrafung. Durch diese Feedbackschleife wird das Verhalten des Agenten Schritt für Schritt angepasst und verbessert.


![image](https://github.com/user-attachments/assets/11fa8f4b-bf7c-458a-a2f7-620d31529b92)
 
*Abbildung 10: Grundprinzip des Reinforcement Learning - Quelle: DatabaseCamp*

**Ablauf:**
1. Agent beobachtet Zustand \( s \)
2. Wählt Aktion \( a \) gemäß seiner aktuellen Policy
3. Umgebung liefert neue Beobachtung \( s' \) und Belohnung \( r \)
4. Policy wird anhand der erhaltenen Belohnung angepasst

---

## 5.2 Q-Learning

Ein grundlegender Algorithmus im RL ist das **Q-Learning**. Dabei wird eine sogenannte **Q-Tabelle** geführt, in der jeder Zustand \( s \) und jede mögliche Aktion \( a \) einen **Q-Wert** zugeordnet bekommt. Dieser gibt an, wie lohnenswert es ist, im Zustand \( s \) die Aktion \( a \) auszuführen.

### Q-Update-Regel:

![image](https://github.com/user-attachments/assets/b86108b5-7327-4528-9563-51c85249ac74)


-  α:  Lernrate (wie stark neue Informationen zählen)  
-  γ: Diskontierungsfaktor für zukünftige Belohnungen  
-  R: unmittelbare Belohnung  
-  s′: Folgezustand

---

### Beispiele:

#### 🐭 Maus im Labyrinth
Eine Maus lernt über viele Versuche, wo sich Käse (+1) und Fallen (–1) befinden. Die Q-Werte helfen ihr, sich künftig für den besten Weg zu entscheiden.

#### 🤖 Roboter im Raum
Ein Roboter erhält Belohnung nur im Zielraum F. Mithilfe von Q-Learning lernt er, welchen Pfad er bevorzugen sollte – den mit dem höchsten erwarteten Q-Wert.

---

## Deep Q-Learning

Wenn der Zustandsraum sehr groß oder kontinuierlich ist, wird die Q-Tabelle unpraktikabel. Hier kommt **Deep Q-Learning (DQN)** zum Einsatz: Ein neuronales Netz approximiert die Q-Funktion.

Vorteile:
- Spart Speicherplatz
- Ermöglicht Einsatz bei komplexen Aufgaben, z. B.:
  - Spiele wie Breakout oder Pong
  - Bewegungssteuerung physischer Roboter

> DQN wurde 2015 durch DeepMind bekannt und zeigte, dass ein Agent auf menschlichem Niveau Atari-Spiele spielen kann (Mnih et al., 2015).

---

## Nützliche Tools und Ressourcen

- 🧪 **OpenAI Gym:** Simulationsumgebung für RL-Experimente  
  → https://gym.openai.com/  
- 🧠 **MushroomRL:** Python-Bibliothek mit vielen RL-Algorithmen  
  → https://mushroomrl.readthedocs.io/en/latest/  
- 🎥 **YouTube-Tutorialreihe (deutsch):**  
  [Teil 1](https://www.youtube.com/watch?v=pc-H4vyg2L4) – [Teil 2](https://www.youtube.com/watch?v=0ODB_DvMiDI) – [Teil 3](https://www.youtube.com/watch?v=7cF3VzP5EDI) – [Teil 4](https://www.youtube.com/watch?v=Wypc1a-1ZYA)

---

## Quellen

1. DatabaseCamp – *Q-Learning einfach erklärt*  
   [https://databasecamp.de/en/ml/q-learnings](https://databasecamp.de/en/ml/q-learnings)  
2. Vorlesungsfolien TH Augsburg – *Reinforcement Learning*  
   - 380_RL-2.pdf  
   - 381_RL_SideScroller-2.pdf  
3. Mnih et al. (2015): *Human-level control through deep reinforcement learning*,  
   Nature, Vol. 518, [DOI: 10.1038/nature14236](https://doi.org/10.1038/nature14236)  
4. Wikipedia – [Reinforcement Learning (englisch)](https://en.wikipedia.org/wiki/Reinforcement_learning)

