# Holo (Windows / Python port)

This is a Python port of the original macOS **Holo** app: it listens through your
microphone, detects taps around the four zones next to your laptop, classifies
which zone was tapped, and runs an assigned action.

It's a rewrite, not a wrapper — the onset detector, the impact-vs-speech gate,
the passive tap-acoustics feature extractor, and the ridge-regression +
nearest-neighbor zone classifier are ported line-for-line from the Swift
original (`HoloCore`) so the classification behavior should match closely.

## What's different from the macOS version

- **Passive sensing only.** The macOS app's Active/Hybrid modes emit an
  ultrasonic chirp through the MacBook speakers and correlate the echo — that's
  speaker/mic-pair specific and wasn't ported. Passive tap acoustics (the
  default mode, and the one the README recommends) is fully implemented.
- **Actions adapted for Windows:** "Run Shortcut" (macOS Shortcuts) is replaced
  with "Run script" (`.bat`/`.ps1`/`.exe`), "Open Application/Item" uses
  `os.startfile`, and clipboard/speech/screenshot use `pyperclip` / `pyttsx3` /
  `Pillow`+`pywin32`.
- **No sandboxing, entitlements, or Core Audio route checks** — those are
  macOS/App-Store concepts with no Windows equivalent. Any input device you
  select will work; there's no "must be the built-in mic" restriction.
- **No diagnostics/sensing-comparison screen, no 60-tap guided evaluation
  screen, no debug WAV capture.** The GUI covers calibration, live detection,
  actions, and profile management — the parts of the app people actually use
  day to day. These could be added back in if you want them.
- GUI is Tkinter instead of SwiftUI, since that ships with Python and needs no
  extra install on Windows.

## Setup (Windows)

1. Install Python 3.10+ from python.org (check "Add to PATH" during install).
2. Open a terminal in this folder and install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run it:
   ```
   python main.py
   ```

On first run, Windows may prompt for microphone access — allow it.

## Using it

1. **Settings** — pick your microphone from the input device list.
2. **Desk → Start Listening** — arms the mic. (Detection runs, but actions
   only fire once "Desk active" is also checked.)
3. **Calibration → Begin Calibration** — walks you through 10 taps per zone,
   40 total, same as the original: tap naturally around the highlighted zone,
   pause briefly between taps. Undo removes the last tap; Cancel discards the
   session.
4. **Save Profile** once all 40 are collected — trains the classifier and
   reports leave-one-out accuracy, then saves to
   `%APPDATA%\Holo\Profiles\<id>.json`.
5. **Actions** — assign one action per zone (visual only, copy text, speak
   text, open website, run script, open app/file, run shell command,
   screenshot to clipboard). Each row has an inline Test button.
6. **Profiles** — load or delete saved desk profiles later.

## Known limitations (carried over from the original, still true here)

- A profile is specific to one laptop, desk, and laptop position. Moving the
  laptop invalidates it — recalibrate.
- Typing, talking, and nearby impacts can resemble taps; the impact/sustained
  gate reduces false positives but can't eliminate them.
- Classification accuracy depends on consistent, natural taps during
  calibration and on your desk being reasonably rigid (solid wood, laminate,
  engineered wood work best; glass/metal/soft surfaces are unreliable).
- This hasn't been measured on real hardware in this environment — treat it as
  a working prototype and validate the leave-one-out accuracy shown after
  calibration before relying on it.

## Project layout

```
holo/
  zone.py            DeskZone, RejectionReason, ZoneActionKind
  signal_models.py   SignalQuality, TapFeatureVector, LabeledTap, DetectedTap
  gate.py            impact-vs-sustained-sound gate
  detector.py         streaming onset detector (adaptive noise floor, pre-roll)
  features.py        passive tap-acoustics feature extraction (30 features)
  classifier.py       ridge-regression + nearest-neighbor zone classifier
  profile.py          profile persistence (JSON, %APPDATA%\Holo)
  actions.py           Windows action dispatch
  capture.py            sounddevice microphone capture -> detector
  app_state.py          controller wiring it all together for the GUI
gui.py                 Tkinter desktop UI
main.py                entry point
requirements.txt
```
