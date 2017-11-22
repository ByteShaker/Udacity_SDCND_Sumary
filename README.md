#Udacity Self-Driving-Car Engineer / Summary

##Term 1:

### Einführung

1. Kurze Einführung.

Vorstellung der beteiligten Spezialisten im Kurs (Sebastian Thrun, Ryan Keenan und David Silver).
Zudem werden die beteiligten Firmen und Ihr Engagement vorgestellt.
Im Anschluss werden die Systeme angesprochen, die State-of-the-Art im Bereich Autonomes Fahren darstellen. 
Zudem wird die Struktur des Kursprogramms kurz vorgestellt.

Bemerkung: für absolute Einsteiger wird inzwischen ein kurzes Kursprogramm "Beginner Self-Driving-Cars" angeboten.
In diesem werden Grundlagen der Programmierung und der Sensoren erläutert.

2. <b>Projekt 1: Erfassen von Fahrspurlinien</b> (Level: Beginner / Codesprache: Python)

Nahezu unmittelbar startet nicht nur eine kurze theoretische Einführung, sondern das erste Projekt muss bearbeitet werden.

Hierbei wird ein Videostream einer Autobahnfahrt zur Verfügung gestellt. 
Die Kursteilnehmer müssen mit Hilfe von einfachen "Computer Vision" Techniken die grobe Position und Ausrichtung der Fahrspurlinien erkennen können.
Verwendete Methoden sind hierbei "Canny Edge" und "Hough Transform".

[![Lane Finding / Videobeispiel](./media/white.mp4)](./media/white.mp4)


### Deep Learning

1. Machine Learning:

In diesem ersten Kapitel werden fundamentale Kenntnisse im Bereich Machine Learning wiederholt.
Dabei werden vor Allem Beispiele aus "Regression" und "Classification" betrachtet.

2. Neural Network:

Im Folgenden wird auf die Prinzipielle Funktionsweise von Perzeptronen eingegangen.
In Anlehnung ein Neuron im menschlichen Gehirn wird Speziell auf Aktivierungsfunktionen eingegangen.
Zudem stehen mehrere kurze Übungen an, in denen die Studenten eigenständig ihr erstes Neuronales Netz in Python strukturieren und programmieren.

3. Logistic Classifier:

Implementierung eines "Logistic Classifiers" in TensorFlow (Python) zur Erkennung von handschriftlichen Buchstaben und Zahlen.
Trainieren des Modells an Hand eines klassifizierten Datensatzes.

4. Optimization:

Untersuchen von Optimierungstechniken für bessere Performance der Classifier.
Dies schließt vor Allem die wichtigem Parameter "Validation-Set", "Test-Set", "Gradient Descent", "Momentum" und "Learning-Rates" mit ein.
Diese Parameter und ihre Auswirkungen werden später anhand von Codebeispielen und Projekten verdeutlicht.

5. Rectified Linear Units:

In diesem Abschnitt werden die Effekte verschiedener Aktvierungsfunktionen auf die Leistung und Güte des Modells erforscht.

6. Regulaization:

Es wird auf verschiedene Techniken eingegangen, um die Eigenschaften eines Neuronalen Netzes gegenüber Over- und Underfitting zu verbessern.
Dabei ist vor Allem "Dropout" eine Technik, die im Anschluss anhand eines Codebeispiels vermittelt wird.

7. Convolutional Neural Networks:

Um für das nächste Projekt bestmöglich vorbereitet zu sein, werden die Bausteine eines "Convolutional Neural Networks" erklärt.
Insbesondere "Filter", "Stride" und "Pooling" erläutert und mit einfachen Codebeispielen das Verständnis der Studenten abgefragt.

8. <b>Projekt 2: Klassifizierung von Verkehrsschildern</b> (Level: Intermediate / Codesprache: Python, TensorFlow)

In Verwendung der Inhalte aus den vorangehenden Kapiteln ist es Aufgabe der Studenten eigenständig ein Convolutional Neural Network zu entwerfen, das in der Lage ist Bilder von Strassenschildern zu klassifizieren.
Für die Bearbeitung werden Quellen von verschiedenen Papern genannt, die einfache Architekturen benennen, mit Hilfe derer eine robuste Lösung konstrukiert werden kann.
Explizites Augenmerk liegt darauf, dass die Netze weder over- noch underfitted werden.

[![Traffic Sign Classification / Beispiel einer Implementierung](https://github.com/ByteShaker/sdcnd_traffic-signs/blob/master/Traffic_Signs_Recognition.ipynb)](https://github.com/ByteShaker/sdcnd_traffic-signs/blob/master/Traffic_Signs_Recognition.ipynb)

9. Keras:

Im folgenden wird auf die umfangreiche Keras Bibliothek einegangen, die es mit einfachen Schritten ermöglicht komplexe, vielschichtige Convolutional Neural Networks aufzubauen.
Die Einfachheit von Keras wird der Flexibiltät von TensorFlow gegenübergestellt.

10. Transfer Learning:

Dieses Kapitel behandelt die Vorgehensweise, um bereits gelernte bestehende Netzwerke auf ähnliche Zwecke anzuwenden. 
Dies spart in der Regel lange Trainingszeiten. Jedoch muss darauf geachtet werden die neuen Aufgabenstellungen sinnvoll in das Netzwerk einzuarbeiten.
Verwendete bestehende kanonische Netzwerke sind: AlexNet, VGG, GoogLeNet und ResNet.

11. <b>Projekt 3: Kopieren von menschlichem Fahrverhalten </b> (Level: Expert / Codesprache: Pyhon, Keras)

Aufgabe der Studenten ist es ihr Fahrverhalten in einem Fahrsimulator durch ein Neuronales Netz zu kopieren. 
Dazu sammeln Sie Bilddaten und parallel dazu ihre Steuerungsbefehle während Sie das Fahrzeug auf der simulierten Rennstrecke bewegen. 
Mit Hilfe dieser selbstgenierten Trainingsdaten starten Sie ein Neuronals Netz zu trainieren.
Das entsprechende Netz müssen Sie selber planen und die entscheidenden Tuning-Parameter bestimmen.
Gelingt der Prozess lernt das Fahrzeug die Steuerbefehle des Nutzers anhand aktueller Bilder selbstständig zu wiederholen.
Da das Netz nicht blind kopiert, gelingt es bei ausreichender Datenmenge temporäre Fehler des menschlichen Fahrers zu glätten und so effizienter und robuster die Rennstrecke zu absolvieren.

[![Behaviour Cloning / Beispiel einer Implementierung](https://github.com/ByteShaker/behaviour_cloning)](https://github.com/ByteShaker/behaviour_cloning)

### Computer Vision

1. Cameras:

In diesem Kapitel wird auf die notwendige Physic von Kameralinsen eingegangen. 
Zudem wird erläutert welche Techniken notwendig sind, um Bildmaterial für eine maschinelle Verarbeitung vorzubereiten.
Im Speziellen wird auf Kamerakalibrierung, Bildendzerrung und das Transoformieren von Bildperspektiven eingegangen.

2. Lane Finding:

Dieses Kapitel beschäftigt sich mit der Weiterentwicklung des Spurerkenners aus dem ersten Projekt. 
Dazu werden fortgeschrittene Methoden erläutert, um auch bei kurvigen Strassen, schlechtem Wetter oder ungünstigen Lichtverhältnissen eine robuste Erkennung zu gewährleisten.

3. <b>Projekt 4: Weiterentwicklung Spurerkennung</b> (Level: Expert / Codesprache: Python, OpenCV)

In diesem Projekt werden erneut Videostreams verschiedener Autobahnfahrten bzw. Landstrassen bearbeitet.
Im Gegensatz zu den Bildern aus Projekt 1 befinden wir uns diesmal auf kurvigen Autobahnabschnitte und haben mit unterschiedlichen Fahrbahnbelägen und Lichtverhältnissen zu kämpfen.
Durch Kalibrieren der Kamera und Transformieren der Bilder in den "Birds-Eye-View" werden die Videos für die weitere Verarbeitung vorbereitet.
Im Anschluss wird ein Verfahren implementiert, mit dem Spurlinienpunkte erkannt werden und daraus per Polynomischem Fit die entsprechenden "Splines" berechnet werden.
Dies beinhaltet die Anwendung verschiedener Filter und die Kombination der richtigen Extraktionen.
Am Ende des Projekts sind die Studenten in der Lage die berechnete Fahrspur ins Realbild zu transformieren und entsprechende Kurvenparameter zu berechnen.

[![Advanced Lane Finding / Beispiel einer Implementierung und Videobeispiel](https://github.com/ByteShaker/advanced-lane-lines/blob/master/writeup_template.md)](https://github.com/ByteShaker/advanced-lane-lines/blob/master/writeup_template.md)

4. Support Vector Machines:

Im Vergleich zu den bisher behandelten Neuronalen Netzen wird in diesem Kapitel die Funktionsweise von SVMs erklärt und anhand einer kurzen Programmieraufgabe verdeutlicht.
Dies dient dazu auch weitere Machine Learning Algorithmen kennenzulernen, die für spezielle Anwendungen performanter sein können als Neuronale Netze.

5. Decision Tree:

Ähnlich der SVMs wird in diesem Kapitel die Funktionsweise von Entschiedungsbäumen erläutert. 
Auch diese werden in einem Programmierbeispiel dazu verwendet Bilder zu klassifizieren.

6. Histogram of Oriented Gradients:

Ein wichtiges und effizientes Feature zur Klassifizierung von Bildern stellt das sog. HOG-Feature dar.
Hierbei werden Muster der Gradientenlinien im Bild identifiziert und dazu genutzt wiederkehrende Objekte zu erkennen.

7. Deep Neural Network:

Hier wird eine weitere DNN-Architektur behandelt und anhand von Codebeispielen auf die unterschiedliche Perfomance im Vergleich zu SVM, Decisin Tree und HOG eingegangen.
Eines oder mehrere dieser Technologien wird je nach Gefühl des Studenten für das anstehende Projekt verwendet.

8. Vehicle Detection:

Um für das finale Projekt des ersten Abschnitts vorbereitet zu sein, wird in diesem Kapitel das Lokalisieren von Fahrzeugen im Kamerabild erläutert.
Dazu werden verschiedene Inhalte der vorhergehenden Kapitel vereint. Zudem wird auf verschiedene Strategien des Subsampelns eines Bildabschnittes eingegangen.

9. <b>Project 5 : Vehicle Tracking</b> (Level: Expert / Codesprache: Python, SciPy, OpenCV)

Das letzte Projekt des ersten Terms beinhaltet das Erkennen und Tracken von Fahrzeugen in einem Videostram.
Hierzu müssen die Studenten verschiedene der oben beschriebenen Methoden anwenden um eine robuste Erkennung zu gewährleisten.
Zielsetzung ist jedoch nicht nur die robuste Erkennung der Fahrzeuge pro Frame, sondern das Predicieren der erkannten Objekte in den nächsten Frame, um eine Trennung der Fahrzeuge bei Überlappung gewährleisten zu können.

[![Vehicle Tracking / Beispiel einer Implementierung und Videobeispiel](https://github.com/diyjac/SDC-P5)](https://github.com/diyjac/SDC-P5)

##Term 2:
