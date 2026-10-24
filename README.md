# Meinungsbildung: wer bestimmt das Ergebnis? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-meinungsbildung-demo.streamlit.app/)** -->

Neuntes Stück der **Netzwerkanalyse-Reihe** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ und Wurzel des **Anpassungs-Arms** (Stücke 9 bis 12): in den Ausbreitungs-Stücken 5 bis 8 und 13 ändert sich der Zustand eines Betriebs nur in eine Richtung, hier **gleichen sich die Betriebe an** und können zurückkehren, sich zersplittern oder erstarren. Die [Zentralitäts-Demo](https://sebastianhanisch-centrality-demo.streamlit.app/) kennt den PageRank als stationäre Verteilung einer Markov-Kette; hier ist dieselbe Gewichtung (ungedämpft) das Gewicht, mit dem ein Betrieb in den Konsens eingeht. Die [Blockmodell-Demo](https://sebastianhanisch-stochastic-block-demo.streamlit.app/) liefert die Verbünde als Engpass.

In einem Netz von Subunternehmern und Lieferanten schätzt jeder Betrieb eine Zahl (zum Beispiel „wie lange dauert die Umstellung auf das neue Verfahren“, eine Zahl zwischen 0 und 1) und gleicht sie laufend mit der seiner Partner ab. Die Demo stellt das **DeGroot-Modell** (jeder ersetzt seine Meinung durch den Mittelwert seiner Partner) dem **Friedkin–Johnsen-Modell** (ein Teil der Anfangsmeinung bleibt: Sturheit) gegenüber und zeigt, **wo das Netz landet, wie schnell, wer den Wert bestimmt und wie viel Vielfalt bleibt**.

## Kernfrage

**Wo landet das Netz, wie schnell, und wer bestimmt den Wert?** Der Hook: der Konsens ist **kein Mittelwert der Köpfe, sondern ein mit dem Grad gewichteter**; wenige Hubs bestimmen den Wert, und die „Weisheit der Menge“ schrumpft entsprechend. Dazu: Verbünde sind der Engpass für die Geschwindigkeit, und schon ein kleiner Anteil Sturheit lässt einen Rest der Vielfalt stehen (bei 5 % Sturheit bleiben 6,8 bis 13,3 % der Anfangsstreuung), in Verbünden besonders. Alle Ergebnisse sind **Modellbefunde**, keine Aussagen über die Wirklichkeit.

## Befunde und Korrekturen gegenüber dem Plan

- **Der Konsens ist der gradgewichtete Mittelwert** (Vermutung im Gesamtplan bestätigt): π<sub>*i*</sub> = *d*<sub>*i*</sub> / Σ *d*<sub>*j*</sub> ist der linke Eigenvektor von *W*. In der Messreihe weicht der Endzustand der Iteration bei Selbstgewicht 50 % um höchstens 1,45 · 10⁻⁸ vom Wert π · *x*(0) ab (größte Abweichung in „Verbünde“; Abbruch bei Spanne unter 10⁻⁶), und ein Orakel prüft den Konsens gegen den Eigenvektor aus `numpy.linalg.eig` auf 120 Zufallsnetzen.
- **Wenige Hubs bestimmen das Ergebnis:** haben nur die zehn Betriebe mit den meisten Partnern die Meinung 1 (alle anderen 0), landet der Konsens im Potenzgesetz-Netz je nach Netz bei etwa 18 bis 30 % (Median der acht festen Netze: 28,9 %; unabhängige Nachprüfung auf 100 weiteren Netzen: Median 23,7 %, 10-bis-90-%-Bereich 18,2 bis 30,3 %) und im skalenfreien Netz bei 19,9 %, bei zehn zufälligen Betrieben im Mittel bei 4,0 %. Im Ring (5,1 %) und im Gitter (4,3 %) ist der Unterschied klein.
- **Die Weisheit der Menge schrumpft:** im Potenzgesetz-Netz zählt der Konsens je nach Netz nur wie etwa 50 bis 130 unabhängige Köpfe von 250 (Median der acht festen Netze der Demo: 56,7; Fehlervarianz ×4,41 gegenüber dem einfachen Mittelwert; Simulation ×4,21), im skalenfreien Netz wie 125,4 (×1,99), im Ring wie 248,0 (×1,01). **Die Potenzgesetz-Netze streuen stark:** zwischen den acht festen Netzen liegt *n*<sub>eff</sub> zwischen 46,0 und 103,2. Die Vormessung im Einzelplan (andere Netze) fand dort 100 von 250, eine unabhängige Nachprüfung auf 200 weiteren Netzen desselben Generators einen Median von 90,0 (10-bis-90-%-Bereich 52,6 bis 131,3): die Größenordnung ist ein Median über wenige Netze, kein fester Wert.
- **Verbünde sind der Engpass:** bei Selbstgewicht 50 % und 240 Betrieben braucht die Mittelung bis zur Spanne 10⁻³ 38,5 Schritte, wenn 50 % der Kanten innerhalb der Verbünde liegen, und 2893,5 Schritte bei 99 %.
- **Ring und Gitter sind langsam:** bei Selbstgewicht 50 % 951,5 bzw. 2397 Schritte gegen 83 im Zufallsnetz (|λ₂| 0,996 gegen 0,865).
- **Ein bipartites Netz pendelt:** das Gitter mit vier Partnern hat bei Selbstgewicht 0 den Eigenwert −1; nach 2000 Schritten ist die Spanne noch 0,041. Mit 10 % Selbstgewicht konvergiert es (1161 Schritte), und das Gitter mit acht Partnern (Diagonalen, nicht bipartit) konvergiert auch ohne (680 Schritte).
- **Jede Sturheit hält Vielfalt (aber nur einen Teil):** bei Empfänglichkeit 95 % bleiben im Zufallsnetz 6,8 % der Anfangsstreuung erhalten, im Ring 13,3 % und im Gitter 12,3 %: langsame Netze halten mehr.
- **Verbünde halten die Spaltung:** bei Empfänglichkeit 95 % bleiben von einer Spaltung in drei gegen drei Verbünde 12,3 % der Anfangsstreuung erhalten, wenn 50 % der Kanten innerhalb der Verbünde liegen, und 85,5 % bei 99 %.
- **Technischer Befund (BLAS-Threads):** auf einem ausgelasteten Rechner braucht das Lösen eines 250 × 250-Gleichungssystems mit mehreren BLAS-Threads etwa 300 ms, mit einem Thread unter 1 ms (gemessen am 2026-10-20 auf dem Entwicklungsrechner); die Module setzen die Thread-Zahl vor dem ersten Import von numpy auf 1.

## Modell

Netz ungerichtet, jeder Betrieb gewichtet seine Partner gleich. **DeGroot:** *x*(*t* + 1) = *W* *x*(*t*) mit *W* = *s* *I* + (1 − *s*) *D*<sup>−1</sup>*A* (Selbstgewicht *s*, Rest gleichmäßig auf die Partner). **Konsens:** *x*<sub>∞</sub> = Σ π<sub>*i*</sub> *x*<sub>*i*</sub>(0) mit π<sub>*i*</sub> = *d*<sub>*i*</sub> / Σ *d*<sub>*j*</sub>; Geschwindigkeit über |λ₂|. **Friedkin–Johnsen:** *x*(*t* + 1) = Λ *W* *x*(*t*) + (*I* − Λ) *x*(0) mit der Empfänglichkeit Λ = diag(*a*) (Sturheit 1 − *a*), Gleichgewicht *x*\* = (*I* − Λ *W*)<sup>−1</sup> (*I* − Λ) *x*(0) aus einem linearen Gleichungssystem. **Weisheit der Menge** (Golub und Jackson 2010): Fehlervarianz des Konsenses σ² Σ π<sub>*i*</sub>², effektive Zahl unabhängiger Köpfe *n*<sub>eff</sub> = 1 / Σ π<sub>*i*</sub>². Ausgewertet wird die größte zusammenhängende Gruppe.

## Methodik

| Baustein | Inhalt |
|---|---|
| **Netze** | Generatoren aus `k-core-demo` (Stück 1) als Kopie: Zufallsnetz, Potenzgesetz-Netz, skalenfrei (BA), Verbünde (80 % der Kanten innen), Ring mit Abkürzungen, Gitter; für die Verbünde-Experimente ein eigenes Blockmodell mit einstellbarem Anteil der Kanten innen |
| **Hauptansicht** | Netz mit Meinungen als Farbe, Größe = Gewicht im Konsens, Regler „Schritt“, Verlauf der Spanne (logarithmisch), Kennzahlen (Konsens, Schritte, \|λ₂\|, *n*<sub>eff</sub>, übrige Streuung) |
| **Experiment 1** | Konsenswert bei Meinung 1 in zehn Betrieben: die mit den meisten Partnern, zehn zufällige (Erwartungswert exakt 10 / *m*), die mit den wenigsten |
| **Experiment 2** | Schritte bis Spanne unter 10⁻⁶ und \|λ₂\| je Netztyp bei Selbstgewicht 0 und 50 %; Verbünde über den Anteil innen (Spanne unter 10⁻³); das bipartite Gitter |
| **Experiment 3** | *n*<sub>eff</sub> und Fehlervarianz je Netztyp, Formel gegen Simulation (1 000 Wiederholungen je Netz) |
| **Experiment 4** | Streuung des Gleichgewichts über die Empfänglichkeit 50, 80, 90, 95, 99 % je Netztyp |
| **Experiment 5** | Spaltung in drei gegen drei Verbünde: Streuung des Gleichgewichts über den Anteil innen (50 bis 99 %) und die Empfänglichkeit 80, 95, 99 % |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, *n* = 250 (das Gitter hat 256 Betriebe), *c* = 6, Median über die Netze; Verbünde-Experimente mit 240 Betrieben und *c* = 6; Anfangsmeinungen gleichverteilt in [0, 1] (Experiment 5: +1 und −1).

| Frage | Befund |
|---|---|
| **Stimmen die Bausteine?** | Konsens gegen den linken Eigenvektor auf 120 Zufallsnetzen; *W*<sup>*t*</sup> gegen die Zeilen π<sup>T</sup>; das Gleichgewicht aus dem Gleichungssystem gegen 10 000 Iterationsschritte und gegen die Neumann-Reihe (auch mit je Betrieb verschiedener Empfänglichkeit); \|λ₂\| gegen die symmetrische Form und gegen die gemessene Abklingrate der Spanne; Eigenwert −1 genau bei bipartiten Netzen (`networkx.is_bipartite`); Fehlervarianz σ² Σ π² gegen eine Simulation mit 40 000 Läufen (`tests/test_oracle_meinungsbildung.py`). Auf den 12 Netzen der Messreihe weicht das Gleichgewicht von 10 000 Iterationen um weniger als 10⁻¹² ab, die Zeilensummen von *W* um weniger als 10⁻¹⁴ von 1, und das Gleichgewicht liegt im Bereich der Anfangsmeinungen |
| **Wo landet das Netz, wie schnell?** | Tabelle zu Experiment 2 |
| **Wer bestimmt den Wert?** | Tabelle zu Experiment 1 |
| **Wie viel ist die Menge wert?** | Tabelle zu Experiment 3 |
| **Wie viel Vielfalt bleibt?** | Tabellen zu Experiment 4 und 5 |
| **Standardlauf** | 150 Betriebe, 450 Partnerschaften; Konsens nach 40 Schritten; \|λ₂\| 0,725; Konsenswert (Gradgewicht) 0,469, einfacher Mittelwert 0,470; *n*<sub>eff</sub> 132,2; Gewicht der zehn größten Hubs 11,4 %; übrige Streuung 0,0 % |
| **Preset Hubs bestimmen das Ergebnis** | 150 Betriebe, 444 Partnerschaften; Konsens nach 37 Schritten; \|λ₂\| 0,712; Konsenswert (Gradgewicht) 0,233, einfacher Mittelwert 0,067; *n*<sub>eff</sub> 91,0; Gewicht der zehn größten Hubs 23,3 %; übrige Streuung 0,0 % |
| **Preset Gitter: der Konsens braucht Zeit** | 144 Betriebe, 506 Partnerschaften; Konsens nach 395 Schritten; \|λ₂\| 0,971; Konsenswert (Gradgewicht) 0,478, einfacher Mittelwert 0,473; *n*<sub>eff</sub> 137,7; Gewicht der zehn größten Hubs 7,9 %; übrige Streuung 0,0 % |
| **Preset Gitter ohne Selbstvertrauen** | 144 Betriebe, 264 Partnerschaften; pendelt, kein Konsens nach 2000 Schritten; \|λ₂\| 1,000; Konsenswert (Gradgewicht) 0,477, einfacher Mittelwert 0,473; *n*<sub>eff</sub> 141,1; Gewicht der zehn größten Hubs 7,6 %; übrige Streuung 8,5 %; Spanne nach 2000 Schritten 0,048 |
| **Preset Sturheit: Vielfalt bleibt** | 150 Betriebe, 450 Partnerschaften; Gleichgewicht nach 32 Schritten; \|λ₂\| 0,725; Konsenswert (Gradgewicht) 0,469, einfacher Mittelwert 0,470; *n*<sub>eff</sub> 132,2; Gewicht der zehn größten Hubs 11,4 %; übrige Streuung 6,9 % |
| **Preset Spaltung entlang der Verbünde** | 149 Betriebe, 450 Partnerschaften; Gleichgewicht nach 69 Schritten; \|λ₂\| 0,881; Konsenswert (Gradgewicht) 0,470, einfacher Mittelwert 0,503; *n*<sub>eff</sub> 130,1; Gewicht der zehn größten Hubs 11,9 %; übrige Streuung 21,5 % |

**Experiment 2: Geschwindigkeit je Netztyp** (Median über 8 Netze; Schritte bis die Spanne unter 10⁻⁶ fällt; letzte Spalte: größte Abweichung des Endzustands vom Wert π · *x*(0) bei Selbstgewicht 50 %):

| Netztyp | \|λ₂\| bei *s* = 0 | Schritte bei *s* = 0 | \|λ₂\| bei *s* = 50 % | Schritte bei *s* = 50 % | Abweichung vom Konsens |
|---|---|---|---|---|---|
| Zufallsnetz | 0,735 | 40 | 0,865 | 83 | 5,57 · 10⁻⁹ |
| Potenzgesetz-Netz | 0,794 | 55,5 | 0,896 | 112 | 7,84 · 10⁻⁹ |
| Skalenfreies Netz (BA) | 0,716 | 36 | 0,856 | 76,5 | 5,24 · 10⁻⁹ |
| Verbünde | 0,879 | 79,5 | 0,939 | 163,5 | 1,45 · 10⁻⁸ |
| Ring mit Abkürzungen | 0,977 | 473,5 | 0,988 | 951,5 | 2,48 · 10⁻⁹ |
| Gitter | 0,992 | 1196,5 | 0,996 | 2397 | 2,41 · 10⁻¹¹ |

**Verbünde: Schritte bis die Spanne unter 10⁻³ fällt** (240 Betriebe, *c* = 6, Selbstgewicht 50 %):

| Kanten innerhalb der Verbünde | \|λ₂\| | Schritte |
|---|---|---|
| 50 % | 0,878 | 38,5 |
| 80 % | 0,935 | 69,5 |
| 90 % | 0,973 | 163,5 |
| 95 % | 0,988 | 307 |
| 99 % | 0,999 | 2893,5 |

**Das Gitter ohne Selbstgewicht:**

| Fall | Selbstgewicht | kleinster Eigenwert | Ergebnis |
|---|---|---|---|
| Gitter mit 4 Partnern | 0 % | −1,000 | pendelt: Spanne 0,041 nach 2000 Schritten |
| Gitter mit 4 Partnern | 10 % | −0,800 | konvergiert nach 1161 Schritten |
| Gitter mit 8 Partnern | 0 % | −0,518 | konvergiert nach 680 Schritten |

**Experiment 1: Wer bestimmt das Ergebnis?** (Konsenswert in Prozent, wenn zehn Betriebe die Meinung 1 haben und alle anderen 0):

| Netztyp | die 10 mit den meisten Partnern | 10 zufällige (Erwartungswert) | die 10 mit den wenigsten Partnern |
|---|---|---|---|
| Zufallsnetz | 7,7 | 4,0 | 1,2 |
| Potenzgesetz-Netz | 28,9 | 4,0 | 1,3 |
| Skalenfreies Netz (BA) | 19,9 | 4,0 | 2,0 |
| Verbünde | 7,6 | 4,0 | 1,1 |
| Ring mit Abkürzungen | 5,1 | 4,0 | 3,8 |
| Gitter | 4,3 | 3,9 | 2,4 |

**Experiment 3: Weisheit der Menge** (Anfangsmeinung = Wahrheit + unabhängiges Rauschen; Fehlervarianz des Konsenses gegenüber dem einfachen Mittelwert aller Köpfe):

| Netztyp | Betriebe | *n*<sub>eff</sub> | Faktor (Formel *m* · Σ π²) | Faktor (Simulation) | größtes Gewicht π<sub>max</sub> |
|---|---|---|---|---|---|
| Zufallsnetz | 249 | 215,7 | ×1,15 | ×1,16 | 0,90 % |
| Potenzgesetz-Netz | 250 | 56,7 | ×4,41 | ×4,21 | 8,93 % |
| Skalenfreies Netz (BA) | 250 | 125,4 | ×1,99 | ×2,02 | 3,46 % |
| Verbünde | 250 | 217,4 | ×1,15 | ×1,18 | 0,87 % |
| Ring mit Abkürzungen | 250 | 248,0 | ×1,01 | ×1,01 | 0,57 % |
| Gitter | 256 | 249,2 | ×1,03 | ×1,05 | 0,43 % |

**Experiment 4: übrige Streuung des Gleichgewichts** (Prozent der Anfangsstreuung; Selbstgewicht 0; Empfänglichkeit *a*):

| Netztyp | *a* = 50 % | 80 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|
| Zufallsnetz | 53,8 | 24,5 | 13,1 | 6,8 | 1,4 |
| Potenzgesetz-Netz | 54,2 | 25,7 | 14,2 | 7,6 | 1,6 |
| Skalenfreies Netz (BA) | 53,2 | 24,0 | 12,7 | 6,6 | 1,4 |
| Verbünde | 53,4 | 24,4 | 13,3 | 7,1 | 1,5 |
| Ring mit Abkürzungen | 54,1 | 29,4 | 19,8 | 13,3 | 4,1 |
| Gitter | 54,3 | 29,0 | 18,9 | 12,3 | 4,4 |

**Experiment 5: übrige Spaltung in Verbünden** (Prozent der Anfangsstreuung; drei Verbünde bei +1, drei bei −1):

| Kanten innerhalb der Verbünde | *a* = 80 % | 95 % | 99 % |
|---|---|---|---|
| 50 % | 38,1 | 12,3 | 2,7 |
| 80 % | 56,9 | 22,9 | 5,5 |
| 90 % | 75,4 | 41,0 | 12,1 |
| 95 % | 85,2 | 56,3 | 21,2 |
| 99 % | 96,3 | 85,5 | 54,7 |

### Hypothesen des Einzelplans

| | Hypothese | Ergebnis |
|---|---|---|
| H1 | Der Konsens ist der mit dem Grad gewichtete Mittelwert der Anfangsmeinungen, nicht der einfache | ✅ Abweichung höchstens 1,45 · 10⁻⁸ in jedem Netztyp; Orakel gegen den Eigenvektor |
| H2 | Die Geschwindigkeit hängt an \|λ₂\|: Ring und Gitter sind langsam, Verbünde sind ein Engpass | ✅ Gitter 2397 gegen Zufallsnetz 83 Schritte; Verbünde mit 99 % Kanten innen 2893,5 Schritte |
| H3 | Auf einem bipartiten Netz konvergiert die reine Partner-Mittelung nicht (Eigenwert −1) | ✅ Gitter mit 4 Partnern pendelt; schon 10 % Selbstgewicht genügen |
| H4 | „Weisheit der Menge“: die Fehlervarianz ist σ² Σ π<sub>*i*</sub>², Hubs verschwenden Genauigkeit | ✅ Potenzgesetz *n*<sub>eff</sub> = 56,7 von 250 (×4,41), Simulation und Formel im Band; stark netzabhängig (46,0 bis 103,2) |
| H5 | Wenige Hubs bestimmen das Ergebnis | ✅ die zehn größten Hubs bewegen den Konsens um 28,9 % (Potenzgesetz, acht feste Netze; je nach Netz etwa 18 bis 30 %) bzw. 19,9 % (skalenfrei) gegen 4,0 % bei zufälliger Wahl; nicht im Ring und im Gitter |
| H6 | Jede Sturheit hält Vielfalt, das Gleichgewicht ist ein lineares Gleichungssystem | ✅ übrige Streuung bei *a* = 95 %: 6,8 % bis 13,3 %; Gleichgewicht gegen 10 000 Iterationsschritte |
| H7 | Verbünde halten Vielfalt: die Spaltung entlang der Gemeinschaften bleibt | ✅ bei *a* = 95 %: 12,3 % bei 50 % Kanten innen, 85,5 % bei 99 % |
| H8 | Das Gleichgewicht liegt zwischen der kleinsten und der größten Anfangsmeinung | ✅ auf allen 12 Netzen der Messreihe und in den Orakel-Tests |

## Ehrliche Grenzen

- **Gleiches Gewicht je Partner, ungerichtetes Netz:** mit unterschiedlichem Vertrauen oder einseitigen Beziehungen ist das Gewicht im Konsens nicht mehr der Grad, sondern der linke Eigenvektor der neuen Matrix. Das Ergebnis „Hubs bestimmen den Wert“ hängt an dieser Annahme.
- **Selbstgewicht als Kunstgriff:** ohne es konvergiert die reine Partner-Mittelung auf bipartiten Netzen nicht; mit ihm läuft die Einigung langsamer (bei 50 % etwa doppelt so viele Schritte).
- **Gleiche Empfänglichkeit für alle:** die Experimente setzen dieselbe Sturheit für alle Betriebe; verschieden sture Betriebe sind nur als Rechnung im Kern (und als Orakel-Test) geprüft, in der App nicht einstellbar.
- **Stichproben:** Mediane über acht (App: vier) feste Netze; die Streuung zwischen den Netzen ist nur für *n*<sub>eff</sub> im Potenzgesetz-Netz ausgewiesen (46,0 bis 103,2) und dort groß. Zufällige Anfangsmeinungen je Netz: ein Strom, keine Wiederholungen.
- **Größte Gruppe:** isolierte Betriebe und kleine Inseln bleiben unberücksichtigt (im Potenzgesetz-Netz und in den Verbünden bei kleinem *c* spürbar).
- **Dichte Rechnung:** bis 250 Betriebe (Gitter 256); die Eigenwerte kommen aus der dichten Rechnung, die Iteration hat eine Obergrenze von 30 000 Schritten (bei pendelnden Netzen 2 000).
- **Modellbefund:** die Demo zeigt Verhalten im Modell, nicht die Wirkung in der Wirklichkeit. Wie weit Meinungsmodelle echte Daten treffen, ist offen (Flache u. a. 2017). Der Zusammenhang zu PageRank (π ist die stationäre Verteilung der ungedämpften Kette) ist Einordnung, nicht gemessen.
- **Literatur:** DeGroot, J. Am. Stat. Assoc. 69(345), 1974, S. 118–121; Friedkin und Johnsen, J. Math. Sociol. 15(3–4), 1990, S. 193–205; Golub und Jackson, Am. Econ. J.: Microeconomics 2(1), 2010, S. 112–149; Flache u. a., JASSS 20(4), 2017 – nachgeschlagen am 2026-10-20; die Arbeiten selbst sind nicht vollständig gelesen.

## Verifikation

- **Kern:** Handwerte auf Mini-Instanzen (`tests/conftest.py`, `tests/test_algorithm.py`): Pfad (π = ¼, ½, ¼, Konsens ¼), zwei Betriebe (ohne Selbstgewicht vertauschen sie die Meinungen, mit ½ Konsens ½), Dreieck, Stern, Friedkin–Johnsen mit zwei Betrieben (*x*\* = (⅔, ⅓)), Fehlerfälle der Übergangsmatrix, Bilder des Schritt-Reglers, Anfangsmeinungen je Modus.
- **Orakel:** `tests/test_oracle_meinungsbildung.py` – Konsens gegen den Eigenvektor, *W*<sup>*t*</sup> gegen π<sup>T</sup>, Friedkin–Johnsen gegen Iteration und Reihe, |λ₂| gegen die symmetrische Form und die Abklingrate, Eigenwert −1 gegen die Bipartitheit, Fehlervarianz gegen Simulation, Invarianten (Zeilensummen, Nichtnegativität, Meinungen im Bereich, π · *x*(*t*) erhalten).
- **Netze und Läufe:** gültig in allen Ecken der Regler, deterministisch je Seed (auch über Prozesse mit verschiedenem Hash-Seed), Blockmodell mit dem verlangten Anteil innen (`tests/test_scenario.py`).
- **Verdikte:** jede Schwelle und jede Zahl im Text an von Hand gesetzten Werten (`tests/test_verdicts.py`), Randfälle (Regler-Ecken für alle Netztypen und Modi, Obergrenze der Schritte, kleinste Netze in allen Experimenten, alle Abbildungen) in `tests/test_edge_cases.py`.
- **Befunde:** jede Zahl dieser README in `tests/test_claims.py`.
- **Mutationsprüfung des Kerns:** (`mbd_algorithm.py`, 94 Stellen, ohne die langsamen Aussagen-Tests in der Suite der Überlebenden): 18 Stellen sind durch wortgleiche Zeilen in Nachbar-Demos gesichert, 74 Mutanten gefangen (9 davon erst nach Ergänzung von Tests: Fehlermeldungen von `fj_equilibrium` samt vollständig empfänglichem Betrieb, ein einzelner Iterationsschritt, Netze mit zwei Betrieben und `spectrum_summary`), 2 als äquivalent festgehalten (Untergrenze 0 → −1 bei einer Schrittliste, die nie negativ wird; `<` gegen `<=` bei einer Zufallszahl gegen eine Wahrscheinlichkeit). Zustand in `tests/mutation_state.json`.
- **Presets:** vollständig, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung, jedes Preset zeigt, wofür es benannt ist.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jeder Netztyp und Modus, Schritt-Regler, neuer Lauf, Permalink, alle fünf Experimente) und Footer-Test.
- **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf (`tests/test_layout.py`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `mbd_algorithm.py` | Kern: Übergangsmatrix, Konsens, DeGroot-Lauf, Eigenwerte, Friedkin–Johnsen (Gleichungssystem und Iteration), effektive Kopfzahl, Anfangsmeinungen, Blockmodell |
| `mbd_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `mbd_evaluation.py` | Analyse, Verdikt-Logik, Messreihen über feste Netze |
| `mbd_layout.py` | Fester Kopf (Stylesheet) |
| `mbd_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `mbd_scenario.py` | Netze: Generatoren (Kopie aus `k-core-demo`), Anordnung |
| `mbd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx, scipy (nur für die Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Instanzen und Handwerte, Orakel, Netze, Verdikte, Randfälle, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe, Befunde |

## Bewusst nicht umgesetzt

- **Verschiedene Empfänglichkeit je Betrieb, gewichtete und gerichtete Netze:** eigene Fragen; im Kern vorbereitet (Vektor *a*), in der App nicht einstellbar.
- **Zufällige Übernahme statt Mittelung** (Voter-Modell, Stück 10), **begrenztes Vertrauen** (Stück 11), **Standortwahl nach Nachbarn** (Stück 12): eigene Demos der Reihe.
- **Meinungen in mehreren Dimensionen** und **Daten aus der Wirklichkeit:** nicht Teil der Demo.
- **Netze über 250 Betriebe live:** die Eigenwerte kommen aus der dichten Rechnung.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
