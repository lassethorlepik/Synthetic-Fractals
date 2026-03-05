# Synthetic Fractals

GPU-driven animated fractal renderer with optional music-reactive color modulation and frame recording.

![Fractal Preview](assets/preview.gif)

## Setup

### 1. Prerequisites

- Python `3.10+` (tested on Python `3.13`)
- A GPU/driver setup that supports OpenGL for `moderngl`
- Includes `tkinter` by default (used for file pickers)

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install pyglet moderngl pillow numpy
```

Optional CUDA acceleration for export backends:

```powershell
python -m pip install cupy-cuda12x
```

If both `CUDA_HOME` and `CUDA_PATH` are set, make sure they point to the same CUDA version.

## Run

```powershell
python app.py
```

Common options:

```powershell
python app.py --seed captures/echo_fissure.json
python app.py --audio song.wav
python app.py --fullscreen --hide-debug
python app.py --record --audio song.wav --seed captures/echo_fissure.json
```

`--record` writes frames to `recordings/<seed>_<timestamp>/`.

## Controls

- `R`: generate a new fractal seed
- `M`: choose/load a WAV file for music-reactive playback
- `Up/Down`: increase/decrease music sensitivity
- `Space`: pause/resume camera zoom
- `A`: toggle fractal animation
- `S`: save current seed JSON to `captures/`
- `L`: load seed JSON from file chooser
- `D`: toggle debug overlay
- `F`: toggle fullscreen
- `P`: save a frame to `captures/`
- `B`: cycle export backend (`cupy`, `numpy`, `python`)
- `Esc`: exit fullscreen or close window

## Benchmark

```powershell
python app.py --benchmark --width 1280 --height 720 --rounds 3
```

## Notes

- Live rendering uses `pyglet` + `moderngl` shader rendering.
