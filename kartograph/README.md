# Kartograph

Handyvideo rein, begehbare 3D-Szene raus — lokal auf dem eigenen Rechner, ohne
Cloud. Du filmst einen Raum mit dem Handy, das Video landet über WLAN auf dem
Notebook, dort rekonstruiert [LingBot-Map][lingbot] die Geometrie, und im
Browser drehst du das Ergebnis herum.

```
  Handy  ──WLAN──▶  Notebook                              ──▶  Browser
  filmt             LingBot-Map → Punktwolke → GLB             drehen, zoomen
```

## Läuft das auf einem Mac?

Ja — aber die offizielle Anleitung von LingBot-Map sagt etwas anderes, deshalb
hier die Begründung. Dort stehen CUDA 12.8, FlashInfer und NVIDIA Kaolin als
Voraussetzungen. Auf Apple Silicon gibt es nichts davon. Trotzdem läuft die
Rekonstruktion, weil diese drei Dinge gar nicht am Modell hängen:

| Angebliche Voraussetzung | Tatsächlich | Was Kartograph tut |
|---|---|---|
| FlashInfer | Import steckt in `try/except`, das Modell hat durchgehend ein `use_sdpa`-Flag | schaltet außerhalb von CUDA auf PyTorch-SDPA |
| NVIDIA Kaolin | kommt im Kernpaket `lingbot_map/` **kein einziges Mal** vor, nur im Video-Renderer `demo_render/` | Renderer wird nicht benutzt |
| CUDA-Extensions | gehören zum selben Video-Renderer | dito |
| feste `.cuda()`-Aufrufe | nur in `compute_distance_matrix_flow`, das **nirgends aufgerufen** wird | irrelevant |
| `torch.cuda.empty_cache()` | überall mit `torch.cuda.is_available()` abgesichert | greift einfach nicht |

Übrig bleibt das Kernpaket, und dessen Abhängigkeiten sind reines Python:
Pillow, einops, safetensors, opencv, scipy. `demo.py` wählt allerdings fest
zwischen CUDA und CPU — MPS kommt dort nicht vor. Ohne Zutun liefe die Inferenz
auf einem Mac also auf der CPU, ohne dass es jemand merkt.
`kartograph/device.py` trifft die Wahl deshalb selbst.

**Was dabei wegfällt:** das gerenderte Kamerafahrt-Video aus `demo_render`. Das
braucht Kaolin und ist an CUDA gebunden. Stattdessen gibt es den interaktiven
Browser-Viewer, in dem du dich ohnehin frei bewegen kannst.

> **Ehrlich gesagt:** Der MPS-Pfad ist aus dem Quelltext hergeleitet, aber nie
> auf echter Hardware gelaufen — hier stand kein Mac und keine GPU zur
> Verfügung. Die Rückprojektion und die Weboberfläche sind getestet (14 Tests),
> die Inferenz auf Apple-GPU ist es nicht. Falls MPS bei einer Operation
> aussteigt, fängt `PYTORCH_ENABLE_MPS_FALLBACK=1` das auf der CPU ab; im
> Zweifel `KARTOGRAPH_DEVICE=cpu` setzen und vergleichen.

## Einrichten

```bash
cd kartograph
./setup.sh
```

Das legt eine virtuelle Umgebung an, klont LingBot-Map, lädt den Checkpoint
(4,6 GB) und holt three.js. Beim ersten Mal dauert es zehn bis zwanzig Minuten,
fast alles davon Download. Der Aufruf ist wiederholbar — was schon da ist, wird
übersprungen.

## Benutzen

```bash
source .venv/bin/activate
python -m kartograph
```

Dann <http://localhost:8765> öffnen. Auf der Seite steht ein QR-Code; den mit
der Handykamera scannen, und das Handy landet direkt auf der Aufnahmeseite.
Beide Geräte müssen im selben WLAN sein.

Video aufnehmen, hochladen, fertig — den Rest kannst du am Notebook verfolgen.
Ein Video lässt sich auch direkt am Notebook ins Fenster ziehen.

### Gute Aufnahmen

- **Langsam gehen.** Bewegungsunschärfe ist der häufigste Grund für matschige Ergebnisse.
- **Quer halten**, mehr Blickfeld pro Bild.
- **20 bis 60 Sekunden** reichen für einen Raum.
- **Einmal im Kreis** und am Startpunkt enden: das Modell erkennt die Schleife und begradigt damit die ganze Szene.
- **Licht an.** Dunkle Ecken werden zu Löchern.

## Was dabei herauskommt

Pro Szene unter `data/scenes/<id>/`:

| Datei | Inhalt |
|---|---|
| `points.bin` | Punktwolke im Kartograph-Format, das der Viewer lädt |
| `scene.glb` | dieselbe Wolke als glTF — für Blender, Preview.app, CloudCompare |
| `predictions.npz` | Rohausgabe: Tiefen, Kameraposen, Konfidenzen |
| `inference.json` | Gerät, Laufzeit, Sekunden pro Frame |

## Stellschrauben

Alles über Umgebungsvariablen, nichts davon muss gesetzt werden:

| Variable | Standard | Wirkung |
|---|---|---|
| `KARTOGRAPH_DEVICE` | automatisch | `mps`, `cuda` oder `cpu` erzwingen |
| `KARTOGRAPH_FPS` | `4` | Frames pro Sekunde, die aus dem Video genommen werden |
| `KARTOGRAPH_MAX_FRAMES` | `240` | Obergrenze; mehr Frames heißt länger warten |
| `KARTOGRAPH_VIEWER_POINTS` | `3000000` | Punktbudget für den Browser |
| `KARTOGRAPH_PORT` | `8765` | Port |
| `LINGBOT_MODEL` | `models/lingbot-map.pt` | anderer Checkpoint, z.B. `lingbot-map-long.pt` für große Szenen |

## Wenn etwas klemmt

**Handy erreicht das Notebook nicht.** Gleiches WLAN? Manche Gastnetze und
Hotel-WLANs isolieren Geräte voneinander, dann hilft nur ein anderes Netz oder
ein Hotspot. Die macOS-Firewall muss eingehende Verbindungen für Python
erlauben.

**„Modell fehlt".** `./setup.sh` erneut laufen lassen, der Download wird
fortgesetzt.

**Speicherfehler bei der Rekonstruktion.** `KARTOGRAPH_MAX_FRAMES=120` setzen.
36 GB reichen normalerweise gut, aber ein zweiminütiges Video bei 8 fps sind
knapp tausend Frames.

**Viewer bleibt schwarz.** Dann fehlt three.js unter
`kartograph/static/vendor/`. `./setup.sh` holt es nach.

**Ergebnis steht auf dem Kopf oder ist spiegelverkehrt.** Die Konvention steckt
in `FLIP` in `viewer.js`; die Tests in `tests/test_export.py` prüfen die
Rückprojektion gegen von Hand nachgerechnete Werte.

## Tests

```bash
python tests/test_export.py   # Rückprojektion und Binärformat
python tests/test_server.py   # HTTP-Schnittstelle
```

Beide brauchen weder GPU noch Checkpoint.

## Aufbau

```
kartograph/
├── device.py            Geräteauswahl MPS/CUDA/CPU
├── runner_inference.py  Video → predictions.npz   (Unterprozess, braucht die GPU)
├── export_scene.py      NPZ → points.bin + GLB    (reines NumPy)
├── pipeline.py          Warteschlange, ein Job zur Zeit
├── server.py            Upload, Jobs, Auslieferung
└── static/              Oberfläche und three.js-Viewer
```

Warum Unterprozesse: der Webserver bleibt ansprechbar, der Speicher wird nach
jedem Job garantiert freigegeben, und Abbrechen ist ein simples Kill.

Warum ein eigener Exporter statt `demo_render/interactive_viewer/npz_to_glb.py`:
das Skript behauptet im Kopfkommentar, ohne CUDA auszukommen, ruft dann aber
`.cuda()` auf seinen Tensoren auf. Die Rückprojektion ist reine
Kameramathematik, also rechnet Kartograph sie geräteunabhängig selbst.

Warum ein eigenes Binärformat neben GLB: bei Millionen Punkten ist ein
glTF-Parser im Browser reine Wartezeit, wenn die Daten ohnehin als dichtes
Float32-Array vorliegen. Das GLB bleibt trotzdem, weil es überall aufgeht.

## Sicherheit

Der Server bindet an `0.0.0.0` und hat keine Anmeldung — er ist fürs eigene
WLAN gedacht. Wer ihn ins Internet stellt, gehört hinter einen Reverse Proxy
mit Authentifizierung.

## Lizenz

Kartograph steht unter der MIT-Lizenz. [LingBot-Map][lingbot] und die
Checkpoints stehen unter Apache 2.0 und werden nicht mitgeliefert, sondern von
`setup.sh` geholt.

[lingbot]: https://github.com/robbyant/lingbot-map
