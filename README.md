# Garg–Könemann – Näherung ohne LP-Löser – Streamlit-Demo

*(noch nicht deployed)*

Neuntes Stück der **Netzwerkfluss-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", drittes im Mehrgüter-Ast, Kontrast zur [Column Generation](https://github.com/sebastian-hanisch/mcf-column-generation-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Garg–Könemann** (mit der Variante von **Fleischer**) für den Mehrgüterfluss – an einem wachsenden Beispiel.
Die Column Generation löst in jeder Runde ein LP. Garg–Könemann kommt **ohne LP-Löser** aus: jede Kapazitätszeile bekommt eine **Länge**, zu Beginn winzig. Ein Kürzeste-Wege-Lauf (Dijkstra, dasselbe Orakel wie im Pricing) findet den kürzesten Weg der Güter, auf ihn wird die Engpassmenge geschoben, und die Längen der berührten Zeilen wachsen **multiplikativ** um $1+\varepsilon\cdot\text{Anteil}$.
Volle Kanten werden dadurch lang und meiden sich selbst – der Fluss weicht aus. Am Ende ist die Pfadsumme viel zu groß und wird durch die größte Auslastung geteilt: zulässig und beweisbar höchstens einen Faktor $(1-\varepsilon)^2$ vom Optimum entfernt. **Fleischer** spart Kürzeste-Wege-Aufrufe, indem es je Gut in Phasen mit Schwelle schiebt.
Vehikel: das Distributionsnetz der Vorgänger-Demos (Standard, Seed 155), ein **Streckennetz** (Gitter mit Start-Ziel-Aufträgen) und zwei feste Lehrnetze (Preis-Wende, Frachtnetz mit Bruch); optional mit einer **Kostenzeile** (Budget).

**Einordnung in die Reihe (die Kanten des Graphen):** Dieselben Packungszeilen wie im Master der Column Generation (gemeinsame Kapazität je Kante, Gut-Obergrenze je Werks- und Nachfragekante) und dasselbe Orakel (Dijkstra mit Kantenlänge = Zeilenlängen), aber ohne Duallösung: die Längen sind keine Schattenpreise, sondern eine Näherung davon, die sich aus dem Verhalten des Flusses ergibt.
Das Ziel ist die **maximale Lieferung** (die Kosten zählen nur über die optionale Budgetzeile). Das Folgestück setzt an dieser Kostenzeile an: das **Netzwerkdesign mit Fixkosten** (Benders-Zerlegung, Slope Scaling). Bisher gebaut: die ersten elf Stücke.
```
edmonds-karp-demo (Wurzel: Restgraph, Rückkanten, Max-Flow = Min-Cut)                  [gebaut]
  ├─ dinic-demo (viele kürzeste Wege je Phase: Niveaugraph, blockierender Fluss)        [gebaut]
  ├─ push-relabel-demo (kein Weg: Überschüsse schieben, Höhen anheben)                 [gebaut]
  └─ ssp-demo (Kosten: der billigste Weg im Restgraphen, Potenziale)                    [gebaut]
       ├─ cycle-canceling-demo (negative Kreise löschen) → Netzwerksimplex               [gebaut]
       │    (network-flow-demo)                                                          [gebaut als Fall-Demo]
       ├─ cost-scaling-demo (Push-Relabel + ε-Skalierung, das nutzt OR-Tools)           [gebaut]
       └─ multicommodity-demo (mehrere Güter teilen Kapazität: Kanten-LP, Preise)       [gebaut]
            ├─ mcf-column-generation-demo (Pfade als Spalten, Pricing = Dijkstra)       [gebaut]
            ├─ garg-koenemann-demo (Näherung mit Preisen, ohne LP-Löser)                [dieses Stück]
            └─ fixkosten-netzdesign-demo (Fixkosten: Schranke und Schnitte)             [gebaut]
                 ├─ benders-demo (Entwurf im Master, Fluss im Teilproblem)              [gebaut]
                 └─ Slope Scaling (Heuristik für große Netze)                           [geplant]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` belegt: Beispielnetze über ihre Seeds, Verteilungen über feste Netze (Seeds ab 100000, dieselben wie in den Vorgänger-Demos): 40 Netze für die Güte, 20 für die ε- und die Kontrolltabelle. Standard: 3 Güter, 3 Werke, 3 Verteilzentren, 8 Filialen, Netzdichte 60 %, Streuung 50 %, Auslastung 90 %, ε = 0,3, Garg–Könemann; Streckennetz: Gitter 5 × 4, 70 % der Gitterkanten, Kapazität bis 2, Menge bis 4, vier Güter.
Das Verfahren ist deterministisch und braucht keinen LP-Löser; das **Optimum** zum Vergleich rechnet HiGHS (maximale Lieferung, mit Kostenbudget als zusätzliche Zeile). Zahlen stehen gerundet („etwa“), die Tests prüfen Bänder. Die Kopien aus den Vorgänger-Demos sind bewacht: SSP 53 907 durchsuchte Kanten, Kanten-LP 67 von 76 Einheiten für 1548, Column Generation = Kanten-LP.

| Frage | Ergebnis |
|---|---|
| Ist der Fluss zulässig? | ✅ Ja, nach dem Teilen durch die größte Auslastung: Erhaltung je Gut, gemeinsame Kapazität, Gut-Obergrenzen, auch mitten im Lauf an jedem Aufnahmepunkt. Die Pfadsumme selbst überlastet die Kanten im Mittel bis zum **etwa 44-Fachen** ihrer Grenze. |
| Wie gut ist er wirklich? | Distributionsnetz (ε = 0,3): im Mittel etwa **95 %** des Optimums, schlechtestes Netz etwa 90 %; Streckennetz etwa **96 %** (schlechtestes etwa 91 %). Beispielnetz (Seed 155): 66,5 von 67 Einheiten (**99 %**). |
| Und gegen die Garantie? | ✅ Weit besser: Garantie $(1-\varepsilon)^2$ = 25 / 49 / 64 / 81 % für ε = 0,5 / 0,3 / 0,2 / 0,1, tatsächlich etwa **91 / 95 / 97 / 98,6 %**. Mit der Skalierung der Theorie (statt der größten Auslastung) wären es etwa 59 / 77 / 85 / 93 % – die Theorie-Skalierung ist der Schwachpunkt, nicht das Verfahren. |
| Kann man die Güte beweisen, ohne das Optimum zu kennen? | ✅ Ja: für beliebige Längen $y$ gilt OPT ≤ $D(y)/\alpha(y)$ ($\alpha$ = kürzester Weg). Die aus den Längen bewiesene Güte (Fluss / obere Schranke) liegt bei etwa **75 / 88 / 93 / 97 %** (ε = 0,5 / 0,3 / 0,2 / 0,1); die obere Schranke war in allen Netzen und allen Tests gültig. |
| Wie viele Schritte? | Etwa **230 / 780 / 1 850 / 7 660** Schiebeschritte für ε = 0,5 / 0,3 / 0,2 / 0,1 – sie wachsen mit Steigung etwa 2,2 gegen 1/ε (Theorie: 2). Die beweisbare Schranke $R\cdot\log_{1+\varepsilon}((1+\varepsilon)/\delta)$ (etwa 3 000 bei ε = 0,3) wird nur zu etwa einem Viertel ausgenutzt. |
| Was spart Fleischer? | ✅ **2,6-fach weniger Aufrufe** bei gleichen Schritten und gleicher Güte: Beispielnetz 906 statt 2 337, im Mittel etwa 890 statt 2 290 (Streckennetz etwa 2,4-fach). Garg–Könemann fragt in jedem Schritt jedes Gut (Aufrufe = Güter × Schritte), Fleischer im Mittel wenig mehr als einmal. Die bewiesene Güte ist etwas geringer (etwa 84 % gegen 87 %), weil die Schranke nur am Ende einer Phase bekannt ist. |
| Was kosten die Kosten? | ❌ Ohne Budgetzeile maximiert das Verfahren nur die Menge: der Fluss ist fast optimal in der Menge (99,5 %), aber im Beispielnetz etwa **23 % teurer** als das Kostenminimum des LP-Optimums (etwa 1 900 gegen 1 548). Mit Budgetzeile (Seed 155, ε = 0,2): Budget 100 / 90 / 75 / 50 % der Kosten des LP-Optimums – LP-Optimum 67 / 63,3 / 56,3 / 43,0 Einheiten, das Verfahren etwa **96 %** davon, bei genau diesen Kosten. |
| Braucht es die Zutaten? | ✅ Ja (ε = 0,3, 20 Netze): **ohne Skalierung** ist der Fluss etwa das 42-Fache des Optimums und unzulässig (Überlast bis zum etwa 44-Fachen); mit **additiven statt multiplikativen Längen** bricht das Verfahren nach etwa 6 Schritten ab und erreicht etwa 49 % (schlechtestes Netz unter 50 %). |
| Gewinnt es gegen HiGHS oder die Column Generation? | ❌ **Nein**, in keiner gemessenen Größe (volle Gitter 5 × 4 bis 16 × 10, fünf Güter, Fleischer, ε = 0,3): 72 bis 598 Zeilen, 390 bis 980 Schritte, 620 bis 1 330 Kürzeste-Wege-Aufrufe – die Column Generation braucht 26 bis 205, also **etwa 6- bis 24-mal weniger**, und liefert das Optimum. In Sekunden etwa 10- bis 19-mal langsamer als ein HiGHS-Lauf (nur zur Information: Python-Dijkstra gegen einen Solver). Die Schritte wachsen mit der Netzgröße nur langsam (Faktor 2,5 bei mehr als dem Achtfachen der Zeilen), aber jeder Aufruf wird teurer. Güte etwa 95 bis 97 %. |
| Lehrnetze | **Preis-Wende:** 1,96 von 2 Einheiten in etwa 43 Schritten; die Direktstrecke wird lang, der Umweg kommt dran. **Frachtnetz mit Bruch:** 1,42 von 1,5 (95 %) mit drei Pfaden – die Näherung ist wie das LP teilbar. |
| Beispielnetze | Distributionsnetz (Seed 155, 56 Zeilen, ε = 0,3): etwa 780 Schritte, 2 337 Aufrufe, 99 %, bewiesen mindestens 91 %; mit ε = 0,1: etwa 7 480 Schritte, 99,8 %, bewiesen 99,5 % (Garantie 81 %). Streckennetz (Seed 7, 64 Zeilen): etwa 313 Schritte, 6,5 von 7 Einheiten (93 %), bewiesen 88 %. |

## Was nicht funktioniert hat / Vorab-Hypothesen

Vor dem Schreiben der Texte wurde gemessen; einige Vermutungen aus dem Plan stimmten nicht oder nur teilweise:

- **„Garg–Könemann konkurriert mit Column Generation und HiGHS.“** Nein: auf diesen Größen braucht es ein Vielfaches der Kürzeste-Wege-Aufrufe der Column Generation und ist auch in Sekunden langsamer; der Kreuzungspunkt liegt jenseits dessen, was die Demo rechnet. Der Wert des Verfahrens liegt nicht in diesen Größen, sondern dort, wo kein LP-Löser passt – und in der bewiesenen Güte ohne Optimum.
- **„Die Garantie beschreibt, was man bekommt.“** Nein: die Garantie $(1-\varepsilon)^2$ ist ein Worst-Case-Wert, tatsächlich liegt die Güte bei ε = 0,3 bei 95 % statt 49 %. Der größte Teil des Abstands kommt von der **Skalierung der Theorie**: das Teilen durch die größte Auslastung statt durch $\log_{1+\varepsilon}((1+\varepsilon)/\delta)$ hebt die Güte von etwa 77 % auf 95 % – und ist jederzeit zulässig.
- **„Fleischer ist gleich gut zertifiziert.“** Nur teilweise: dieselbe Güte des Flusses, aber die bewiesene Güte ist etwas geringer (etwa 84 % gegen 87 %), weil eine obere Schranke nur nach vollständigen Phasen bewiesen ist.
- **„Additive Längen sind eine schwächere, aber brauchbare Variante.“** Nein: mit additiver Erhöhung greift die Abbruchbedingung nach wenigen Schritten, bevor sich der Fluss verteilt hat – etwa 49 % statt 95 %. (Die erste Fassung der Negativkontrolle erhöhte um einen winzigen Betrag und erreichte die Abbruchbedingung nie – ein Lauf bis zur Schrittobergrenze; der Vergleich braucht eine Erhöhung in der Größenordnung der Kapazitätsanteile.)
- **Fund beim Bau (Zertifikat):** In Fleischers Phasenschema ist die obere Schranke nur nach einer **vollständigen** Phase gültig. Die erste Fassung nahm auch die Schwelle einer abgebrochenen Phase (Abbruch bei $D\ge1$ mitten im Durchlauf der Güter) – die „Schranke“ lag dann unter dem Optimum. Der Test der Schranke gegen das LP-Optimum in allen Netzen fängt das.
- **Abweichungen vom Plan:** Port 8683; kein PDF-Export; die Verteilung der Güte läuft über 40 statt 100 Netze (ε = 0,1 kostet je Netz eine halbe Sekunde); die Negativkontrolle „ohne Skalierung“ und die additive Variante sind eine Tabelle plus Regler „Variante“; Sekunden nur zur Information.

## Was die Demo zeigt

- **Aufnahmepunkte:** Regler über die Schritte (die ersten acht einzeln, dann in wachsenden Abständen; der letzte Punkt ist das Ende mit der Skalierung). Links das Netz: je Gut eine Farbe (Breite ~ zulässig skalierter Fluss), orange Unterlage nach dem **Längenfaktor** der Kanten gegenüber dem Start (in drei logarithmischen Stufen, die fünf längsten mit ihrem Faktor), schwarz gestrichelt der zuletzt geschobene Pfad; im Streckennetz Start (Raute) und Ziel (Stern) je Gut. Rechts die Lieferung gegen die Schiebeschritte (zulässiger Fluss steigt, bewiesene obere Schranke sinkt, gestrichelt das LP-Optimum) und die **Güte** in Prozent des Optimums – tatsächlich und bewiesen – gegen die Garantie. ▶️ spielt alles ab.
- **Wie gut und was kostet es?** Lieferung, bewiesene Güte, Schritte (mit der Schranke), Aufrufe; ein Urteil (zulässig und Güte / ohne Skalierung unzulässig / additive Längen schwach / nichts kommt an) und die Verteilung der Güte über 40 feste Netze.
- **Experimente (🔬, auf Abruf):** ε von 0,5 bis 0,1 (Güte, Theorie-Skalierung, Garantie, bewiesene Güte, Schritte, Schranke, Aufrufe beider Verfahren); Kostenbudget 100 bis 50 % gegen das LP mit Budgetzeile; Größe (Aufrufe gegen die Column Generation, Sekunden zur Information); Negativkontrollen.
- **Optionen:** ε (0,5 / 0,3 / 0,2 / 0,1), Verfahren, Kostenbudget, Variante. **Feste Netze** (Preis-Wende, Frachtnetz mit Bruch) und zufällige Distributions- und Streckennetze; **Wo die Annahmen enden:** Ganzzahligkeit, Kosten als Ziel, Güte gegen Optimum, Größe, Fixkosten, Zeit.

## Modell und Verfahren

- **Packungs-LP:** $\max\sum_p x_p$ unter $\sum_p a_{rp}x_p\le b_r$; Zeilen: gemeinsame Kapazität je gemeinsamer Kante, Gut-Obergrenze je (Gut, Kante) mit endlicher Grenze (= die Zeilen des Masters der Column Generation), optional eine Kostenzeile (Koeffizient = Kosten des Pfades).
- **Garg–Könemann:** $\delta=(1+\varepsilon)((1+\varepsilon)R)^{-1/\varepsilon}$, $y_r=\delta/b_r$; solange $D(y)=\sum_r b_ry_r<1$: kürzester Weg über alle Güter (ein Dijkstra je Gut), Engpassmenge schieben, $y_r\leftarrow y_r(1+\varepsilon f a_{rp}/b_r)$. Skalierung durch die größte Auslastung (zulässig) bzw. die Theorie-Skalierung $\log_{1+\varepsilon}((1+\varepsilon)/\delta)$ (garantiert $(1-\varepsilon)^2$; im Test gegen das LP geprüft).
- **Fleischer:** Phasen mit Schwelle $\alpha(1+\varepsilon)$, je Gut so lange auf dem kürzesten Weg schieben, wie er unter der Schwelle liegt; nach einer vollständigen Phase eine obere Schranke.
- **Zertifikat:** $\text{OPT}\le D(y)/\alpha(y)$ für beliebige Längen. **Schrittschranke:** jeder Schritt vervielfacht die Länge der Engpasszeile mit $1+\varepsilon$, jede Zeile wächst höchstens $\log_{1+\varepsilon}((1+\varepsilon)/\delta)$-mal, also höchstens $R$ mal so viele Schritte (im Test geprüft).
- **Aufwand:** Schiebeschritte und Kürzeste-Wege-Aufrufe, nie Sekunden.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `gk_constants.py` | Regler-Grenzen, Presets und Hilfetexte, feste Seed-Mengen |
| `gk_presets.py` | Permalink, Preset- und Zufalls-Seed-Logik; ausblendbare Regler werden erst vor dem Zeichnen initialisiert (`seed_widget`) |
| `gk_algorithm.py` | Garg–Könemann und Fleischer, Aufnahmepunkte, Zertifikat, Kostenzeile, Negativkontrollen (additiv, ohne Skalierung) |
| `gk_lp.py` | Kanten-LP für die maximale Lieferung, optional mit Kostenbudget (Vergleichsbasis, HiGHS) |
| `gk_scenario.py`, `gk_model.py`, `gk_paths.py` | Distributionsnetz mit eigenem Zufallsgenerator, Güter, Frachtnetze, Streckennetz, Preis-Netz, Pfade (Kopien der Vorgänger-Demos) |
| `gk_cg.py`, `gk_edge_lp.py`, `gk_ssp.py`, `gk_edmonds_karp.py` | Kopien der Vorgänger: Column Generation (Vergleich der Aufrufe, Master-Zeilen), Kanten-LP, SSP (bewacht) |
| `gk_evaluation.py` | Urteil, Verteilungen, ε-, Kontroll-, Budget- und Größentabelle |
| `gk_visualization.py` | Plotly-Abbildungen (Achsen gesperrt für Touch-Geräte; Kantenbeschriftungen als Annotationen mit heller Hinterlegung; Hover über unsichtbare Marker entlang der Kanten) |
| `tests/` | Algorithmus (Zulässigkeit, gültige obere Schranke, Theorie-Garantie, Schrittschranke, Aufrufe, Budget gegen das LP mit Budgetzeile, Negativkontrollen, Lehrnetze, Aufnahmepunkte), Auswertung, Presets, belegte Zahlen, AppTest-Rauchtests |

Alle Daten sind synthetisch; die Laufzeit braucht numpy, pandas, plotly, streamlit und **scipy** (das Vergleichs-LP, HiGHS), `networkx` ist ein reines Testorakel.

## Lokal starten

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -v
```

Ein Lauf dauert einige Minuten (Verteilungen und Experimente). Schritte und Aufrufe hängen nicht von einem LP-Löser ab; Anteile und Mittel stehen mit Bändern, weil Gleitkomma-Summen auf anderen Plattformen um einen Schritt abweichen können.
Die CI (`.github/workflows/tests.yml`) läuft auf Ubuntu mit Python 3.12, bei jedem Push und wöchentlich mit den jeweils neuesten Bibliotheksversionen.
