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

[![Projekt 1 / Videobeispiel](./media/white.mp4)](./media/white.mp4)


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

9. Keras:

Im folgenden wird auf die umfangreiche Keras Bibliothek einegangen, die es mit einfachen Schritten ermöglicht komplexe, vielschichtige Convolutional Neural Networks aufzubauen.
Die Einfachheit von Keras wird der Flexibiltät von TensorFlow gegenübergestellt.

10. Transfer Learning:

Dieses Kapitel behandelt die Vorgehensweise, um bereits gelernte bestehende Netzwerke auf ähnliche Zwecke anzuwenden. 
Dies spart in der Regel lange Trainingszeiten. Jedoch muss darauf geachtet werden die neuen Aufgabenstellungen sinnvoll in das Netzwerk einzuarbeiten.
Verwendete bestehende kanonische Netzwerke sind: AlexNet, VGG, GoogLeNet und ResNet.

11. <b>Project: Behaviour Cloning </b> (Level: Expert / Codesprache: Pyhon, Keras)

Aufgabe der Studenten ist es ihr Fahrverhalten in einem Fahrsimulator durch ein Neuronales Netz zu kopieren. 
Dazu sammeln Sie Bilddaten und parallel dazu ihre Steuerungsbefehle während Sie das Fahrzeug auf der simulierten Rennstrecke bewegen. 
Mit Hilfe dieser selbstgenierten Trainingsdaten starten Sie ein Neuronals Netz zu trainieren.
Das entsprechende Netz müssen Sie selber planen und die entscheidenden Tuning-Parameter bestimmen.
Gelingt der Prozess lernt das Fahrzeug die Steuerbefehle des Nutzers anhand aktueller Bilder selbstständig zu wiederholen.
Da das Netz nicht blind kopiert, gelingt es bei ausreichender Datenmenge temporäre Fehler des menschlichen Fahrers zu glätten und so effizienter und robuster die Rennstrecke zu absolvieren.

[![Behaviour Cloning / Beispiel einer Implementierung](https://github.com/ByteShaker/behaviour_cloning)](https://github.com/ByteShaker/behaviour_cloning)