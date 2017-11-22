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

11. <b>Projekt 3: Kopieren von menschlichem Fahrverhalten </b> (Level: Advanced / Codesprache: Pyhon, Keras)

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

3. <b>Projekt 4: Weiterentwicklung Spurerkennung</b> (Level: Advanced / Codesprache: Python, OpenCV)

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

9. <b>Project 5: Vehicle Tracking</b> (Level: Expert / Codesprache: Python, SciPy, OpenCV)

Das letzte Projekt des ersten Terms beinhaltet das Erkennen und Tracken von Fahrzeugen in einem Videostram.
Hierzu müssen die Studenten verschiedene der oben beschriebenen Methoden anwenden um eine robuste Erkennung zu gewährleisten.
Zielsetzung ist jedoch nicht nur die robuste Erkennung der Fahrzeuge pro Frame, sondern das Predicieren der erkannten Objekte in den nächsten Frame, um eine Trennung der Fahrzeuge bei Überlappung gewährleisten zu können.

[![Vehicle Tracking / Beispiel einer Implementierung und Videobeispiel](https://github.com/diyjac/SDC-P5)](https://github.com/diyjac/SDC-P5)

##Term 2:

###Sensor Fusion

1. Sensors:

Einleitend in den zweiten Term und das Thema Senor Fusion wird die Physik und Funktionsweise wichtiger Sensoren erläutert.
Im Bereich des autonomen Fahrzeugs sind diese Sensoren Radar und Lidar.

2. Kalman Filters:

Um einen Zustand mit Hilfe verschiedener Datenquellen möglichst genau üebr der Zeit zu verfolgen ist aktuell in der Industrie der Kalman Filter die mathematische Schlüsselkomponente.
In diesem Abschnitt des Nanodegrees implementieren die Studenten selbstständig eine einfache Version in Python. 
Hierbei kombinieren Sie verschiedene aufeinanderfolgende Messungen eines Sensors, um mit der Zeit Gewissheit über den Zustand des Objekts zu erlangen.

3. C++ Primer:

Da alle weiterne Projekte dieses Terms in C++ Implementiert werden müssen, werden an dieser Stelle die grundsätzlichen Konzepte von C++ wiederholt und auf gewisse Stylingregeln eingegangen.

4. <b>Projekt 1: Extended Kalman Filter in C++</b> (Level: Intermediate / Codesprache: C++)

Um nicht lineare Zustandsmodelle mit Hilfe eines Kalman Filters abzubilden existieren Spezialformen dieses Algorithmus.
Eine solche Spezialform ist der Extended Kalman Filter. 
Hierbei werden die nicht linearen Zustände lokal in einen linearen Zusammenhang gebracht und so ein entsprechendes Verhalten angenähert.
Aufgabe in diesem Projekt ist es einen solchen EKF in C++ zu implementieren und Anhand von aufgezeichneten Lidar und Radar Daten die Bewegungen eines Fussgängers zu tracken.

[![Extended Kalman Filter / Beispiel einer Implementierung](https://github.com/NikolasEnt/Extended-Kalman-Filter)](https://github.com/NikolasEnt/Extended-Kalman-Filter)

5. Unscented Kalman Filter:

Im Anschluss an den EKF wird nun der UKF behandelt. 
Dieser Filter Mechanismus, der ebenfalls auf dem Kalman Filter beruht löst nicht lineare Zustandsmodelle nicht durch eine lokale, lineare Annäherung, sondern betrachtet nicht lineare Zusammenhänge mit ihrem genauen Verhalten.
Diese Lösung ist zwar mathematisch sehr anspruchsvoll, jedoch zahlt sich dies in einer besseren Performance als der EKF aus.
Im Hinblick auf die Herausforderungen beim Fusionieren vieler Sensoren und den nichtlinearen Zuständen von Autonomen Fahrzeugen ist dieser Algorithmus klar im Vorteil.

6. <b>Projekt 2: Unscented Kalman Filter in C++</b> (Level: Advanced / Codesprache: C++)

Die Aufgabe in diesem Projekt ist die gleiche wie in dem EKF Projekt. 
Diesmal implementieren die Studenten einen vollständigen UKF um die Lidar und Radardaten zu fusionieren.
Im Anschluss wird die Performance der beiden Filter beim Tracken eines Fussgängers verglichen.

[![Unscented Kalman Filter / Beispiel einer Implementierung](https://github.com/mvirgo/Unscented-Kalman-Filter)](https://github.com/mvirgo/Unscented-Kalman-Filter)

###Localization

1. Motion:

Einleitend in das Kapitel der Lokalisierung wird betrachtet welchen Einfluss Bewegung im Allgemeinen und Bewegungs-Wahrscheinlichkeiten auf die Schätzung des aktuellen Zustands und Orts im globalen Raum haben. 

2. Markov Localization:

Die erste Lokalisierung, die die Studenten in einem Codebesipiel durchfürhen müssen beruht auf einem simplen Bayes Filter.
Hierfür wird ein Vereinfachter Gridraum für die verschiedenen Orte des Fahrzeugs angenommen.

3. Egomotion:

Wie bereits im ersten Kapitel angesprochen hat die Eigenbewegung von Fahrzeugen einen großen Beitrag zur Performance der verschiedenen Modellen.
Hier wird nun auf die verschiedenen physikalischen Bewegungen und ihre Limitationen eingegangen.
Beispielsweise wird das sogenannte "Bicycle"-Modell im Detail betrachtet.
Mit Hilfe einer solchen Bewegungseinschränkung im Wahrscheinlichkeitsraum wird die Lokalisierung eines Fahrzeugs über die Zeit mit Hilfe verschiedener Sensordaten wesentlich robuster.
Auch dieses Kapitel wird mit einem Implementierungsbeispiel abgeschlossen.

4. Particle Filter:

Eine sehr effektive Methode eine Eigenlokalisierung auf Basis verschiedener Sensordaten durchzuführen ist der Partikelfilter.
Dabei kann die Komplexität des globalen Raumes sehr hoch sein. 
Mit Hilfe von Landmarken, die im Kartenmaterial hinterlegt sind wächst über die Zeit die Aufenthaltswahrscheinlichkeit für bestimmte Positionen an.
Dabei werden alle Möglichkeiten so lange im Modell mitgeführt, bis das Ergebnis auf eine einzelne Position mit entsprechender Positionsunsicherheit konvergiert.
Zum besseren Verständnis und zur Anschauung wird eine simple Python Implementierung gefordert.

5. High-Performance Particle Filter:

Nachdem beim Partikelfilter viele Aufenthaltswahrscheinlichekeiten parallel mitgeführt und betrachtet werden ist der Algorithmus dementsprechend rechenintensiv.
Zur Verbesserung der Leistung wird eine Implementierung in C++ empfohlen und hier von den Studenten gefordert.

6. <b>Projekt 3: Kidnapped Vehicle</b> (Level: Advanced / Codesprache: C++)

Der soeben Implementierte Partikelfilter wird nun für das folgende Projekt verwendet und auf Tauglichkeit geprüft.
Hierbei ist es Aufgabe der Studenten ein Fahrzeug, dass irgendwo im global bekannten Raum (Bereich der Karte in dem Landmarken existieren) ausgesetzt wurde, zu lokalisieren.
Hierfür beginnt das Fahrzeug sich zu bewegen und entsprechende Sensordaten rückzumelden.
Über die Zeit soll nun die Schätzung des Ortes auf die tatsächliche Position desfahrzeugs konvergieren.

[![Particle Filter / Beispiel einer Implementierung](http://jeremyshannon.com/2017/06/02/udacity-sdcnd-kidnapped-vehicle.html)](http://jeremyshannon.com/2017/06/02/udacity-sdcnd-kidnapped-vehicle.html)

###Control

1. Control:

Um die Aufgabe und die Notwendigkeit von Steuerungsalgorithemen zu verdeutlichen, wird einleitend in dieses Kapitel die Aktuatorik eines Fahrzeugs erläutert.
Hierbei wird sowohl auf Physikalische Ungenauigkeiten eingegangen, als auch auf einen möglichen Zeitversatz zwischen Detektion und Steuerung.

2. <b>Projekt 4: PID Control</b> (Level: Intermediate / Codesprache: C++)

Eine einfache und klassische Methode eines Steuerungsalgoritmus ist der PID-Regler (Proportional/Integrativ/Derivativ).
Hier werden die verschiedene Bausteine und die jeweilige Aufgabe erläutert.
Im Anschluss implementieren die Studenten eine solche Funktionalität, um ein Fahrzeug über eine bekannte, simulierte Strecke zu steuern.
Hierbei wird sowohl auf den entsprechenden Zeitversatz eingegangen als auch auf Messungenauigkeiten, bzw. Ungenauigkeiten in der Karte.
Die für einen PID-Regler notwendigen Parameter werden mit Hilfe des sogenannten "Twidlle"-Algorithmus bestimmt.

[![PID Controller / Beispiel einer Implementierung und Videobeispiel](https://github.com/ByteShaker/CarND-PID-Control-Project)](https://github.com/ByteShaker/CarND-PID-Control-Project)

3. Linear Quadratic Regulator:

Nachdem der PID-Regler wie auch in dem oben bearbeiteten Projekt wesentliche Schwächen aufweist, wird nun ein anspruchsvolleren Regler vorgestellt.
Dieser Regler ist ersichtlich besser in der Lage das Fahrzeug zu stabilisieren.
Um das zu Bewerkstelligen versucht der Regler in die Zukunft zu blicken und so einen stabileren Zustand herbeizuführen.

4. <b>Projekt 5: Lane Keeping with MPC-Controller</b> (Level: Advanced / Codesprache: C++)

Mithilfe des Model-Predictive-Controller wird wieder das Fahrzeug auf der Strecke gehalten. 
Zudem werden Computer Vision Techniken verwendert um die Fahrspurlinien zu erkennen und somit die Kartendaten zu verbessern.
Hierbei sind die Studenten angehalten und in der Lage wesentlich schneller und stabiler den Rennkurs zu bewältigen.

[![MPC Controller / Beispiel einer Implementierung und Videobeispiel](https://github.com/ByteShaker/CarND-MPC-Project)](https://github.com/ByteShaker/CarND-MPC-Project)


## Term 3:

###Path Planning

1. Environmental Prediction:

Zur Vorhersage des Verhaltens anderer Verkehrsteilnehmer nutzen die Studenten in diesem Kapitel Modellbasierte, Datenbasierte und Hybride Modelle.
Hierbei entscheiden die Modellbasierten Herangehensweisen welches Manöver, aus einer Liste verschiedener expliziter Vorschläge ein Fahrzeug einschlagen wird.
Bei den Datengetrieben Varianten werden Trainingsdaten dazu verwendet wiederkehrende Verhalten vorherzusagen. 
Und die Hybriden Varianten vereinen diese beiden Methoden um noch präziser das Verhalten der anderen Fahrzeuge vorherzusagen.

2. Behaviour Planning:

Zu jedem Zeitpunkt muss ein Pfadplanender Algorithmus eine Entscheidung für das nächste anstehende Eigenverhalten treffen.
Hierzu entwickeln die Studenten sogenannte "Finite State Machines" die zwischen verschiedenen angebotenen expliziten Zuständen wechseln.
Hierbei wird mit Hilfe einer Kostenfunktion entschieden welches dieser Manöver unter aktuellen Bedingenen die geringsten definierten Kosten verursacht.

3. Trajectory Generation:

In diesem Kapitel nutzen die Studenten C++ und die EIGEN-Bibliothek um mögliche Trajektorien für die aktuelle Manöverentscheidung vorzuschlagen.
Auch hierbei wird eine Kostenfunktion genutzt, um zu entscheiden welche der vorgschlagenen Trajektorien besonders sicher, komfortabel und effizient ist.
Diese Trajektorie wird im Anschluss an die Steuerungsalgoritmik des Fahrzeugs übergeben.

4. <b>Projekt 1: Highway Path Planner</b> (Level: Expert / Codesprache: C++)

Alle der vorher besprochenen Themen werden in diesem Projekt vereint.
Es ist Aufgabe der Studenten mit Hilfe eines Autobahnsimulators ein Fahrzeug über die verschiedenen Spuren zu manövrieren und dabei eine besonder effiziente Spurwechselstrategie zu bestimmen.
Je schneller, komfortabler und sicherer das fFhrzeug die Strecke bewältigt desto besser.

[![Highway Path Planner / Beispiel einer Implementierung und Videobeispiel](https://github.com/ByteShaker/CarND-Path-Planning-Project)](https://github.com/ByteShaker/CarND-Path-Planning-Project)


###Elective

In diesem Term kann zwischen zwei Pfaden entschieden werden:

####Elective 1: Advanced Deep Learning mit NVIDIA

1. Fully Convolutional Networks

In diesem Kapitel wird ein "Fully Convolutional Network" gebaut und antrainiert.
Dieses Netzwerk kennzeichnet sich dadurch aus, dass es nicht nur eine Klassifizierung eines Bildes durchführt, sondern eine Klassifizierung auf Pixellevel durchfürht.
Hierbei wird jedes Pixel einem bestimmten Objekt, oder Bereich zugeordnet.
Die Ausgabe ist dementsprechend ebenfalls ein 1x1 Image mit Bereichs bzw. Objektgrenzen.
Hierbei werden drei spezielle Techniken besprochen, um das FCN Modell wirksam zu trainieren: 1x1 Convolution, Upsampling und Layer Skip.

2. Scene Understanding

Der nächste logische Schritt in der Kette ist es, nun nicht nur bestimmte Bereiche oder Objekte zu detektieren, sondern eine Interpretation der aktuellen Szene abzugeben.
Hierzu wird ein Semantic Segmentation Netzwerk aufgebaut, das in der Lage ist über Bounding Boxes hinaus zu funktionieren.
Gestartet wird mit einem kanonischen Netzwerk, wie VGG oder ResNet.
Dabei wird die letzte "fully-connected" Schicht entfernt und durch die drei Techniken aus dem vorherigen Kapitel ergänzt.
Dadurch entsteht eine Strassenszene, die jedes Pixel in der Szene entsprechend Klassifiziert.

3. Inference Optimizations:

Eine der größten Herausforderungen von Semantic Segmentation ist die hohe Anforderung nach paraller Rechenleistung.
Hierzu wird in diesem Kapitel darauf eingegangen, wie mit speziellen Vorgehensweisen und Vereinfachungen eine Beschleunigung des Prozesses herbeigeführt werden kann ohne an zu viel Präzision zu verlieren.
Dabei werden folgende Themen angesprochen: Fusion, Quantization und reduzierte Fließkomma Präzision.

4. <b>Projekt 2: Semantic Segmentation</b> (Level: Advaced / Codesprache: Python, TensorFlow, Keras)

In diesem Projekt am Ende des Electives wird ein Sematic Segmentation Netzwerk implementiert, das in einem Videostream zu jedem Zeitpunkt den freien und befahrbaren Bereich einer Strasse hervorhebt.
Dazu wird sämtliches Wissen verwendet, das in den vorherigen Kapiteln vermittelt wurde.
Als Datenquelle wird KiTTi Road Detection verwendet.

[![Sematic Segmentation / Beispiel einer Implementierung und Videobeispiel](https://github.com/NikolasEnt/Road-Semantic-Segmentation)](https://github.com/NikolasEnt/Road-Semantic-Segmentation)

####Elective 2: Functional Safety mit Elektrobit

1. Introduction:

In diesem Kapitel werden die Studenten mit der ISO 26262 vertraut gemacht.
Diese Norm ist der Standard im Bereich der Funktionalen Sicherheit und Grundlage für die weiteren Themen dieses Electives.

2. Safety Plan:

In diesem Kapitel entwickeln die Studenten einen Sicherheitsplan für einen Spurhalteassistent.
Dabei werden die gleichen Templates verwendet, mit denen Elektrobit Ingeniuere im täglichen Geschäft arbeiten.
Entsprechend dem individuellen Feature wird dieses Template im folgenden ausgefüllt.

3. Hazard Analysis and Risk Management:

Die Studenten vervollständigen in diesem Kapitel eine Gefahren Analyse und eine Risikobewertung für den Spurhalteassistenten.
Teil von diesem HARA ist es Fehler zu prognostizieren, die sich in allen Bereichen des Features erreignen können.
Diese Fehlermeldungen werden in die Funktionssicherheitsanalyse integriert.

4. Functional Safety Concept:

Für jeden Fehler, der im vorhergehenden Kapitel entworfen wurde, wird nun ein Funktionssicherheitskonzept entwickelt.
Dies muss die Anforderung einer anspruchsvolle Performance erfüllen.

5. Technical Safety Concept:

Die Studenten übersetzen nun die Anforderungen an das Funktionssicherheitskonzept in Anforderungen eines technisches Sicherheitskonzept.
Dieses wiederum bewirkt konkrete Rahmenbedingungen des Systems.

6. Software und Hardware:

Funktionssicherheit erfordert entsprechende Regeln für das Implementieren von Hardware und Software.
Hier wird vermittelt wie die Studenten ihr System vor lokalen, temporären und kommunikativen Interferencen schützen können. 
Außerdem wird eine kurze Einführung in MISRA C++ und die wichtigsten Regeln für das Schreiben von C++ Code für Automobilsysteme gegeben.

7. <b>Projekt 2: Safety Case</b> (Level: Intermediate / Codesprache: C++)

In diesem Projekt am Ende des Elektives entwickeln die Studenten mit Hilfe der Richtlinien aus den vorherigen Kapiteln einen ent-to-end Funktionssicherheitsfall für Spurhalteassistenten und die entsprechenden Warnungen.
Beginnend mit einer Risiko und Gefahren Analyse, über weitergehende Dokumentation für funktionale und technische Sicherheitskonzeote, und final der Formulierung der Anforderungen für Hardware und Software.
Diese Aufgaben sind im Bereich der Automobilentwicklung kritisch und befürdern einer expliziten Betrachtung.

[![Functional Safety / Beispiel einer solchen Analyse](https://github.com/Waterfox/CarND-Functional-Safety-Project)](https://github.com/Waterfox/CarND-Functional-Safety-Project)

https://github.com/Waterfox/CarND-Functional-Safety-Project

###Systems Integration

Dies ist das Finale Modul des Nanodegrees. 
Im Unterschied zu allen vorherigen Kapiteln und Projekten ist für das bestehen dieses Abschnittes eine collaborative Bearbeitung notwendig.
Hierzu werden Gruppen von 5 Studenten in einem Featureteam zusammengeführt und erledigen die entsprechenden Aufgaben geschlossen.
Das herunterbrechen des Gesamtprojektes in einzelne Teilaufgaben obliegt dem Team.
Zudem einzigartig ist, dass für das Bestehen nicht nur gefordert ist, dass mit Hilfe der Impementierung des Teams ein Fahrzeug erfolgreich spezielle Aufageben im Simulator bewältigt, sondern der Code wird auf das System von CARLA, dem selbstfahrenden Prototypen von Udacity geflashed und die selben Aufgaben werden in einem realen Umfeld getestet.

1. Vehicle Subsystem:

Um für die Aufgaben gewappnet zu sein werden hier der Systemaufbau von CARLA erläutert. 
Dabei wird auf die Sensoren, die Wahrnehmung, den Planungsalgorithmus und die Steuerungsmechanik eingegangen.

2. ROS and Autoware:

CARLA wird mit zwei der gängisten Open-Souce Bibliotheken betrieben: ROS und Autoware.
In diesem Kapitel werden diese vorgeführt und die Studenten entwerfen ihre ersten ROS nodes und Autoware Module.

3. System Integration:

In diesem Kapitel integrieren die Studenten die entsprechenden ROS Nodes und Autoware Module auf dem Entwicklungssystem von CARLA.
Zudem lernen Sie, wie das transferien des Codes auf das Fahrzeug funktioniert.
Besonders wird nochmals auf die Probleme hingewiesen, die beim Arbeiten mit realer Hardware auftauchen: Wie Zeitverzögerung, Nachrichtenverluste oder Prozessabstürze.

4. <b>Projekt 3: CARLA - System Integrtion</b> (Level: Expert / Codesprache: C++)

Dies ist das finale und entscheidende Projekt des Nanodegrees. 
Die Feature Teams befassen sich mit dem implementieren einer End-to-Lösung, die es CARLA sowohl im Simulator als auch in der Realität ermöglicht auf einer 3-spurigen Strasse zu fahren und auf Lichtsignalanlagen zu reagieren.
Um diese Aufgabe zu bewältigen ist es notwendig alle Inhalte des Nanodegrees aus den vorherigen Projekten in einem ROS-Umfeld zu vereinen.
Einzigen Sensor auf den das Fahrzeug dabei zurückgreift ist eine Kamera, am Fahrzeug montiert ist.
Kontrolliert wird das Fahrzeug lediglich durch Gaspedal/Bremse und das Lenkrad.
Als Backendinformation existiert eine schematische Karte mit niedriger Qualität.

[![CARLA - System Integrtion / Beispiel einer Implementierung](https://github.com/ByteShaker/CarND-Capstone)](https://github.com/ByteShaker/CarND-Capstone)
[![CARLA - System Integrtion / Videobeispiel](https://www.youtube.com/watch?v=1KDDv5UTwig)](https://www.youtube.com/watch?v=1KDDv5UTwig)

