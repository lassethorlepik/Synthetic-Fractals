import argparse
import json
import math
import os
import random
import struct
import time
import wave
from queue import Queue
from threading import Thread
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import moderngl
    import pyglet
    from pyglet.window import key
except ImportError:
    moderngl = None
    pyglet = None
    key = None

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None


WIDTH = 1280
HEIGHT = 720
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
MAX_ITER = 64
TARGET_FPS = 60
OUTPUT_DIR = Path("captures")
RECORDINGS_DIR = Path("recordings")
RECORD_JPEG_QUALITY = 100
RECORD_JPEG_SUBSAMPLING = 0
RECORD_QUEUE_MAX = 50
FULLSCREEN_START_DELAY = 5.0
ANIMATE_FRACTAL = True
ADAPTIVE_DETAIL = False
MIN_ITER = 1024
ITER_GROWTH_PER_OCTAVE = 48
LOCK_LOOP_FOCUS = True
ZOOM_RATE = 0.12
ZOOM_OCTAVES = 18.0
DRIFT_STRENGTH = 0.0
SHADER_AA_STRENGTH = 0.35
MUSIC_SENSITIVITY = 1.0
FOCUS_DEPTH = 10
FOCUS_GRID = 9
FOCUS_ITER = 48
FOCUS_REFINE_PASSES = 3
BOOTSTRAP_GRID = 13
LOOP_LOG_PERIOD = math.log(2.0) * ZOOM_OCTAVES
LOOP_EPSILON = 1e-9
LOOP_TWIST = 0.45
LOOP_JITTER = 0.10
LOOP_JITTER_FREQ = 2.2
LOOP_RADIAL_GAIN = LOOP_LOG_PERIOD / (2.0 * math.tau)
LOOP_RADIAL_HARMONIC = 0.16
SEED_VET_ATTEMPTS = 24
MIN_BOOTSTRAP_SCORE = 2.2
MIN_LOOP_SCORE = 2.0
MIN_LOOP_MOTION = 0.0003
GLSL_ZOOM_RATE = f"{ZOOM_RATE:.8f}"
GLSL_ZOOM_OCTAVES = f"{ZOOM_OCTAVES:.8f}"
CUDA_ZOOM_RATE = f"{ZOOM_RATE:.8f}f"
CUDA_ZOOM_OCTAVES = f"{ZOOM_OCTAVES:.8f}f"
GLSL_LOOP_EPSILON = f"{LOOP_EPSILON:.8e}"
CUDA_LOOP_EPSILON = f"{LOOP_EPSILON:.8e}f"
GLSL_SHADER_AA_STRENGTH = f"{SHADER_AA_STRENGTH:.8f}"


VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 uv;

void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330
uniform float elapsed;
uniform float fractal_elapsed;
uniform int max_iter;
uniform int polynomial_power;
uniform int alt_power;
uniform float blend;
uniform float swirl;
uniform float fold;
uniform float trig_scale;
uniform float drift;
uniform float warp;
uniform float hue_shift;
uniform float zoom_base;
uniform vec2 pan;
uniform float time_scale;

in vec2 uv;
out vec4 fragColor;

vec2 complex_power(vec2 z, int power) {
    if (power == 2) {
        return vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
    }
    if (power == 3) {
        float r2 = z.x * z.x;
        float i2 = z.y * z.y;
        return vec2(z.x * (r2 - 3.0 * i2), z.y * (3.0 * r2 - i2));
    }
    if (power == 4) {
        float r2 = z.x * z.x;
        float i2 = z.y * z.y;
        float ri = z.x * z.y;
        return vec2(r2 * r2 - 6.0 * r2 * i2 + i2 * i2, 4.0 * ri * (r2 - i2));
    }
    vec2 z2 = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
    vec2 z4 = vec2(z2.x * z2.x - z2.y * z2.y, 2.0 * z2.x * z2.y);
    return vec2(z4.x * z.x - z4.y * z.y, z4.x * z.y + z4.y * z.x);
}

vec3 hsv_to_rgb(vec3 hsv) {
    vec3 rgb = clamp(abs(mod(hsv.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    rgb = rgb * rgb * (3.0 - 2.0 * rgb);
    return hsv.z * mix(vec3(1.0), rgb, hsv.y);
}

vec2 loop_map_branch(float angle, float wrapped_log_radius) {
    float twisted_angle = angle
        + wrapped_log_radius * 0.62
        + sin(wrapped_log_radius * 2.7 + 1.6180339) * 0.18;
    float wrapped_radius = exp(wrapped_log_radius);
    return vec2(cos(twisted_angle), sin(twisted_angle)) * wrapped_radius;
}

vec2 loop_map(vec2 p) {
    float radius = length(p);
    float safe_radius = sqrt(radius * radius + __LOOP_EPSILON__ * __LOOP_EPSILON__);
    float angle = atan(p.y, p.x);
    float log_radius = log(safe_radius);
    float phase = log_radius * (6.2831853 / 12.47664925);
    float wrapped_log_radius =
        sin(phase) * 1.98567884
        + sin(phase * 2.0 + 0.7) * 1.98567884 * 0.24;
    vec2 mapped = loop_map_branch(angle, wrapped_log_radius);
    return mapped * (radius / safe_radius);
}

vec3 fractal_color(vec2 c, float time_phase) {
    vec2 z = vec2(
        c.x * 0.35 + sin(c.y * warp + time_phase) * 0.22,
        c.y * 0.35 + cos(c.x * warp - time_phase * 0.7) * 0.22
    );

    float orbit = 0.0;
    float trap = 99.0;
    int iteration = 0;
    float mag2 = dot(z, z);

    for (int i = 0; i < max_iter; i++) {
        float radius = sqrt(max(dot(z, z), 1e-9));
        float angle = atan(z.y, z.x);
        vec2 warped_c = vec2(
            c.x + sin(angle * float(alt_power) + time_phase) * 0.16 * warp,
            c.y + cos(radius * trig_scale - time_phase) * 0.16 * warp
        );

        vec2 primary = complex_power(z, polynomial_power) + warped_c * (0.85 + 0.35 * sin(time_phase + radius * 0.6));
        vec2 secondary = vec2(
            sin(z.x * trig_scale + time_phase),
            cos(z.y * trig_scale - time_phase * 0.8)
        ) * swirl;
        float fold_scale = 1.0 / (1.0 + dot(z, z));
        vec2 folded = vec2(
            sin((z.x * z.y) * fold + time_phase * 0.6),
            cos((z.x - z.y) * fold - time_phase * 0.45)
        ) * fold_scale;

        z = primary * blend + secondary + folded + warped_c * drift;
        mag2 = dot(z, z);
        float radius_next = sqrt(min(mag2, 1e12));
        trap = min(trap, abs(z.x * z.y) + abs(radius_next - 1.0));
        orbit += sin(angle * 3.0 + time_phase) * 0.5 + 0.5;
        iteration = i + 1;
        if (mag2 > 64.0) {
            break;
        }
    }

    float smooth_iter = float(iteration);
    if (mag2 > 1.0 && mag2 < 1e20) {
        smooth_iter = float(iteration) + 1.0 - log2(max(log2(mag2), 1e-6));
    }

    float density = smooth_iter / float(max_iter);
    float trap_glow = exp(-trap * (4.0 + fold * 3.5));
    float orbit_mix = orbit / float(max(iteration, 1));
    float hue = fract(hue_shift + density * 0.45 + trap_glow * 0.28 + orbit_mix * 0.18 + 0.09 * sin(time_phase * 0.41) + 0.05 * cos(time_phase * 0.53));
    float saturation = clamp(0.45 + trap_glow * 0.75 + 0.15 * sin(time_phase + c.x), 0.0, 1.0);
    float value = clamp(density * 0.85 + trap_glow * 0.65 + orbit_mix * 0.25, 0.0, 1.0);
    if (mag2 <= 64.0) {
        value *= 0.16;
        saturation *= 0.25;
    }

    return hsv_to_rgb(vec3(hue, saturation, value));
}

void main() {
    float time_phase = fractal_elapsed * time_scale;
    float zoom_phase = fract(elapsed * __ZOOM_RATE__);
    float zoom_multiplier = exp2(zoom_phase * __ZOOM_OCTAVES__);
    float zoom = zoom_base * zoom_multiplier;
    vec2 local = vec2((uv.x - 0.5) * (4.0 / zoom), (uv.y - 0.5) * (3.0 / zoom));
    vec2 footprint = vec2(fwidth(local.x), fwidth(local.y)) * __SHADER_AA_STRENGTH__;
    vec2 offset_a = vec2(-0.5, -0.5) * footprint;
    vec2 offset_b = vec2(0.5, -0.5) * footprint;
    vec2 offset_c = vec2(-0.5, 0.5) * footprint;
    vec2 offset_d = vec2(0.5, 0.5) * footprint;
    vec3 rgb = (
        fractal_color(loop_map(local + offset_a) + pan, time_phase)
        + fractal_color(loop_map(local + offset_b) + pan, time_phase)
        + fractal_color(loop_map(local + offset_c) + pan, time_phase)
        + fractal_color(loop_map(local + offset_d) + pan, time_phase)
    ) * 0.25;
    fragColor = vec4(rgb, 1.0);
}
""".replace("__ZOOM_RATE__", GLSL_ZOOM_RATE).replace("__ZOOM_OCTAVES__", GLSL_ZOOM_OCTAVES).replace("__LOOP_EPSILON__", GLSL_LOOP_EPSILON).replace("__SHADER_AA_STRENGTH__", GLSL_SHADER_AA_STRENGTH)


CUPY_KERNEL_SOURCE = r"""
extern "C" __global__
void render_fractal(
    unsigned char* out,
    const int width,
    const int height,
    const float elapsed,
    const int max_iter,
    const int polynomial_power,
    const int alt_power,
    const float blend,
    const float swirl,
    const float fold,
    const float trig_scale,
    const float drift,
    const float warp,
    const float hue_shift,
    const float zoom_base,
    const float pan_x,
    const float pan_y,
    const float time_scale
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int pixel_count = width * height;
    if (idx >= pixel_count) {
        return;
    }

    int px = idx % width;
    int py = idx / width;

    float time_phase = elapsed * time_scale;
    float zoom_phase = time_phase * __ZOOM_RATE__;
    zoom_phase = zoom_phase - floorf(zoom_phase);
    float zoom_multiplier = exp2f(zoom_phase * __ZOOM_OCTAVES__);
    float zoom = zoom_base * zoom_multiplier;
    float drift_x = __sinf(time_phase * 0.19f + hue_shift * 6.2831853f) * 0.06f / zoom_multiplier;
    float drift_y = __cosf(time_phase * 0.23f - hue_shift * 4.7123890f) * 0.0444f / zoom_multiplier;
    float local_x = (((float)px / (float)width) - 0.5f) * (4.0f / zoom);
    float local_y = (((float)py / (float)height) - 0.5f) * (3.0f / zoom);
    float local_radius = sqrtf(local_x * local_x + local_y * local_y);
    float safe_radius = sqrtf(local_radius * local_radius + __LOOP_EPSILON__ * __LOOP_EPSILON__);
    float local_angle = atan2f(local_y, local_x);
    float log_radius = logf(safe_radius);
    float phase = log_radius * (6.2831853f / 12.47664925f);
    float wrapped_log_radius =
        __sinf(phase) * 1.98567884f
        + __sinf(phase * 2.0f + 0.7f) * 1.98567884f * 0.24f;
    float twisted_angle = local_angle
        + wrapped_log_radius * 0.62f
        + __sinf(wrapped_log_radius * 2.7f + 1.6180339f) * 0.18f;
    float wrapped_radius = expf(wrapped_log_radius);
    float radial_scale = local_radius / safe_radius;
    float mapped_x = cosf(twisted_angle) * wrapped_radius * radial_scale;
    float mapped_y = sinf(twisted_angle) * wrapped_radius * radial_scale;
    float nx = mapped_x + pan_x + drift_x;
    float ny = mapped_y + pan_y + drift_y;

    float z_real = nx * 0.35f + __sinf(ny * warp + time_phase) * 0.22f;
    float z_imag = ny * 0.35f + __cosf(nx * warp - time_phase * 0.7f) * 0.22f;

    float orbit = 0.0f;
    float trap = 99.0f;
    int iteration = 0;
    float mag2 = z_real * z_real + z_imag * z_imag;

    for (int i = 0; i < max_iter; i++) {
        mag2 = z_real * z_real + z_imag * z_imag;
        float radius = sqrtf(fmaxf(mag2, 1e-9f));
        float angle = atan2f(z_imag, z_real);
        float warped_real = nx + __sinf(angle * alt_power + time_phase) * 0.16f * warp;
        float warped_imag = ny + __cosf(radius * trig_scale - time_phase) * 0.16f * warp;

        float power_real;
        float power_imag;
        if (polynomial_power == 2) {
            power_real = z_real * z_real - z_imag * z_imag;
            power_imag = 2.0f * z_real * z_imag;
        } else if (polynomial_power == 3) {
            float r2 = z_real * z_real;
            float i2 = z_imag * z_imag;
            power_real = z_real * (r2 - 3.0f * i2);
            power_imag = z_imag * (3.0f * r2 - i2);
        } else if (polynomial_power == 4) {
            float r2 = z_real * z_real;
            float i2 = z_imag * z_imag;
            float ri = z_real * z_imag;
            power_real = r2 * r2 - 6.0f * r2 * i2 + i2 * i2;
            power_imag = 4.0f * ri * (r2 - i2);
        } else {
            float z2_real = z_real * z_real - z_imag * z_imag;
            float z2_imag = 2.0f * z_real * z_imag;
            float z4_real = z2_real * z2_real - z2_imag * z2_imag;
            float z4_imag = 2.0f * z2_real * z2_imag;
            power_real = z4_real * z_real - z4_imag * z_imag;
            power_imag = z4_real * z_imag + z4_imag * z_real;
        }

        float primary_scale = 0.85f + 0.35f * __sinf(time_phase + radius * 0.6f);
        float primary_real = power_real + warped_real * primary_scale;
        float primary_imag = power_imag + warped_imag * primary_scale;
        float secondary_real = __sinf(z_real * trig_scale + time_phase) * swirl;
        float secondary_imag = __cosf(z_imag * trig_scale - time_phase * 0.8f) * swirl;
        float fold_scale = 1.0f / (1.0f + mag2);
        float folded_real = __sinf((z_real * z_imag) * fold + time_phase * 0.6f) * fold_scale;
        float folded_imag = __cosf((z_real - z_imag) * fold - time_phase * 0.45f) * fold_scale;

        z_real = primary_real * blend + secondary_real + folded_real + warped_real * drift;
        z_imag = primary_imag * blend + secondary_imag + folded_imag + warped_imag * drift;
        mag2 = z_real * z_real + z_imag * z_imag;

        float radius_next = sqrtf(fminf(mag2, 1e12f));
        trap = fminf(trap, fabsf(z_real * z_imag) + fabsf(radius_next - 1.0f));
        orbit += __sinf(angle * 3.0f + time_phase) * 0.5f + 0.5f;
        iteration = i + 1;
        if (mag2 > 64.0f) {
            break;
        }
    }

    float smooth = (float)iteration;
    if (mag2 > 1.0f && mag2 < 1e20f) {
        smooth = (float)iteration + 1.0f - log2f(fmaxf(log2f(mag2), 1e-6f));
    }

    float density = smooth / (float)max_iter;
    float trap_glow = expf(-trap * (4.0f + fold * 3.5f));
    float orbit_mix = orbit / fmaxf((float)iteration, 1.0f);
    float hue = hue_shift + density * 0.45f + trap_glow * 0.28f + orbit_mix * 0.18f + 0.09f * __sinf(time_phase * 0.41f) + 0.05f * __cosf(time_phase * 0.53f);
    hue = hue - floorf(hue);
    float saturation = fminf(fmaxf(0.45f + trap_glow * 0.75f + 0.15f * __sinf(time_phase + nx), 0.0f), 1.0f);
    float value = fminf(fmaxf(density * 0.85f + trap_glow * 0.65f + orbit_mix * 0.25f, 0.0f), 1.0f);
    if (mag2 <= 64.0f) {
        value *= 0.16f;
        saturation *= 0.25f;
    }

    float hue6 = hue * 6.0f;
    int sector = (int)floorf(hue6);
    float fraction = hue6 - sector;
    float p = value * (1.0f - saturation);
    float q = value * (1.0f - saturation * fraction);
    float t = value * (1.0f - saturation * (1.0f - fraction));

    float red;
    float green;
    float blue;
    switch (sector % 6) {
        case 0: red = value; green = t; blue = p; break;
        case 1: red = q; green = value; blue = p; break;
        case 2: red = p; green = value; blue = t; break;
        case 3: red = p; green = q; blue = value; break;
        case 4: red = t; green = p; blue = value; break;
        default: red = value; green = p; blue = q; break;
    }

    int base = idx * 3;
    out[base] = (unsigned char)(fminf(fmaxf(red, 0.0f), 1.0f) * 255.0f);
    out[base + 1] = (unsigned char)(fminf(fmaxf(green, 0.0f), 1.0f) * 255.0f);
    out[base + 2] = (unsigned char)(fminf(fmaxf(blue, 0.0f), 1.0f) * 255.0f);
}
""".replace("__ZOOM_RATE__", CUDA_ZOOM_RATE).replace("__ZOOM_OCTAVES__", CUDA_ZOOM_OCTAVES).replace("__LOOP_EPSILON__", CUDA_LOOP_EPSILON)


@dataclass
class FractalSeed:
    name: str
    polynomial_power: int
    alt_power: int
    blend: float
    swirl: float
    fold: float
    trig_scale: float
    drift: float
    warp: float
    hue_shift: float
    zoom: float
    pan_x: float
    pan_y: float
    time_scale: float


@dataclass
class AudioReactiveTrack:
    path: Path
    duration: float
    analysis_fps: int
    levels: list[float]
    kicks: list[float]
    smoothed_levels: list[float]
    smoothed_kicks: list[float]
    grooves: list[float]

    @classmethod
    def from_wav(cls, path: str | Path, analysis_fps: int = 120) -> "AudioReactiveTrack":
        track_path = Path(path)
        with wave.open(str(track_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if frame_count <= 0:
            raise ValueError("audio file has no frames")
        if channels <= 0:
            raise ValueError("audio file has no channels")
        if sample_width not in (1, 2, 4):
            raise ValueError("only 8-bit, 16-bit, and 32-bit PCM WAV files are supported")

        sample_total = frame_count * channels
        if sample_width == 1:
            samples = [((value - 128) / 128.0) for value in raw]
        elif sample_width == 2:
            values = struct.unpack(f"<{sample_total}h", raw)
            samples = [value / 32768.0 for value in values]
        else:
            values = struct.unpack(f"<{sample_total}i", raw)
            samples = [value / 2147483648.0 for value in values]

        if channels > 1:
            mono: list[float] = []
            for index in range(0, len(samples), channels):
                mono.append(sum(samples[index : index + channels]) / channels)
        else:
            mono = samples

        # Simple low-frequency band extraction for kick tracking: low-pass at
        # ~180 Hz, then subtract a much slower low-pass around ~35 Hz.
        low_fast: list[float] = []
        low_slow: list[float] = []
        alpha_fast = math.exp(-2.0 * math.pi * 180.0 / sample_rate)
        alpha_slow = math.exp(-2.0 * math.pi * 35.0 / sample_rate)
        state_fast = 0.0
        state_slow = 0.0
        for sample in mono:
            state_fast = state_fast * alpha_fast + sample * (1.0 - alpha_fast)
            state_slow = state_slow * alpha_slow + sample * (1.0 - alpha_slow)
            low_fast.append(state_fast)
            low_slow.append(state_slow)

        kick_band = [fast - slow for fast, slow in zip(low_fast, low_slow)]

        prefix = [0.0]
        band_prefix = [0.0]
        running = 0.0
        band_running = 0.0
        for sample, band_sample in zip(mono, kick_band):
            running += sample * sample
            band_running += band_sample * band_sample
            prefix.append(running)
            band_prefix.append(band_running)

        def rms(prefix_sum: list[float], end_index: int, window_size: int) -> float:
            start_index = max(0, end_index - window_size)
            span = max(1, end_index - start_index)
            energy = prefix_sum[end_index] - prefix_sum[start_index]
            return math.sqrt(max(energy / span, 0.0))

        step = max(1, sample_rate // analysis_fps)
        fast_window = max(64, sample_rate // 40)
        slow_window = max(fast_window * 4, sample_rate // 8)
        levels: list[float] = []
        kick_curve: list[float] = []

        for end_index in range(step, len(mono) + step, step):
            clamped_end = min(len(mono), end_index)
            fast_level = rms(prefix, clamped_end, fast_window)
            slow_level = rms(prefix, clamped_end, slow_window)
            kick_fast = rms(band_prefix, clamped_end, fast_window)
            kick_slow = rms(band_prefix, clamped_end, slow_window)
            level = min(1.0, fast_level * 3.2)
            kick_flux = max(0.0, kick_fast - kick_slow * 1.005)
            kick_ratio = kick_fast / max(fast_level, 1e-6)
            kick_weight = max(0.0, min(1.0, (kick_ratio - 0.34) / 0.38))
            kick_signal = kick_flux * kick_weight
            levels.append(level)
            kick_curve.append(kick_signal)

        kicks: list[float] = [0.0] * len(kick_curve)
        beat_window = max(2, analysis_fps // 7)
        refractory_frames = max(1, int(analysis_fps * 0.06))
        last_peak_index = -refractory_frames
        for index, kick_signal in enumerate(kick_curve):
            start = max(0, index - beat_window)
            recent = kick_curve[start:index]
            local_average = (sum(recent) / len(recent)) if recent else kick_signal
            is_peak = True
            if index > 0 and kick_signal < kick_curve[index - 1]:
                is_peak = False
            if index + 1 < len(kick_curve) and kick_signal < kick_curve[index + 1]:
                is_peak = False
            if index - last_peak_index < refractory_frames:
                is_peak = False
            kick = 0.0
            if is_peak:
                kick = max(0.0, kick_signal - local_average * 1.02) * 34.0
                if kick > 0.0:
                    last_peak_index = index
            kicks[index] = min(1.0, kick)

        smoothed_levels: list[float] = [0.0] * len(levels)
        smoothed_kicks: list[float] = [0.0] * len(kicks)
        grooves: list[float] = [0.0] * len(levels)
        smooth_level = 0.0
        smooth_kick = 0.0
        smooth_groove = 0.0
        for index in range(len(levels)):
            level = levels[index]
            kick = kicks[index]
            if level > smooth_level:
                smooth_level = smooth_level * 0.86 + level * 0.14
            else:
                smooth_level = smooth_level * 0.94 + level * 0.06
            if kick > smooth_kick:
                smooth_kick = smooth_kick * 0.72 + kick * 0.28
            else:
                smooth_kick = smooth_kick * 0.90 + kick * 0.10
            groove_target = smooth_level * 0.78 + smooth_kick * 0.22
            if groove_target > smooth_groove:
                smooth_groove = smooth_groove * 0.84 + groove_target * 0.16
            else:
                smooth_groove = smooth_groove * 0.95 + groove_target * 0.05
            smoothed_levels[index] = smooth_level
            smoothed_kicks[index] = smooth_kick
            grooves[index] = smooth_groove

        duration = frame_count / float(sample_rate)
        return cls(
            path=track_path,
            duration=duration,
            analysis_fps=analysis_fps,
            levels=levels or [0.0],
            kicks=kicks or [0.0],
            smoothed_levels=smoothed_levels or [0.0],
            smoothed_kicks=smoothed_kicks or [0.0],
            grooves=grooves or [0.0],
        )

    def features_at(self, time_seconds: float) -> tuple[float, float, float]:
        if not self.levels:
            return 0.0, 0.0, 0.0
        index = int(max(0.0, time_seconds) * self.analysis_fps)
        index = min(index, len(self.levels) - 1)
        return self.smoothed_levels[index], self.smoothed_kicks[index], self.grooves[index]

def hsv_to_rgb_int(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    if saturation <= 0.0:
        gray = int(value * 255.0)
        return gray, gray, gray
    hue = (hue % 1.0) * 6.0
    sector = int(hue)
    fraction = hue - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    if sector == 0:
        red, green, blue = value, t, p
    elif sector == 1:
        red, green, blue = q, value, p
    elif sector == 2:
        red, green, blue = p, value, t
    elif sector == 3:
        red, green, blue = p, q, value
    elif sector == 4:
        red, green, blue = t, p, value
    else:
        red, green, blue = value, p, q
    return int(red * 255.0), int(green * 255.0), int(blue * 255.0)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def smoothstep_range(edge0: float, edge1: float, value: float) -> float:
    if edge1 == edge0:
        return 1.0 if value >= edge1 else 0.0
    return smoothstep((value - edge0) / (edge1 - edge0))


def loop_map_point(x: float, y: float) -> tuple[float, float]:
    radius = math.hypot(x, y)
    safe_radius = math.hypot(radius, LOOP_EPSILON)
    angle = math.atan2(y, x)
    log_radius = math.log(safe_radius)
    phase = log_radius * (math.tau / LOOP_LOG_PERIOD)
    wrapped_log_radius = (
        math.sin(phase) * LOOP_RADIAL_GAIN
        + math.sin(phase * 2.0 + 0.7) * LOOP_RADIAL_GAIN * LOOP_RADIAL_HARMONIC
    )
    twisted_angle = angle + wrapped_log_radius * LOOP_TWIST + math.sin(wrapped_log_radius * LOOP_JITTER_FREQ + 1.6180339) * LOOP_JITTER
    wrapped_radius = math.exp(wrapped_log_radius)
    radial_scale = radius / safe_radius
    return math.cos(twisted_angle) * wrapped_radius * radial_scale, math.sin(twisted_angle) * wrapped_radius * radial_scale


def loop_map_arrays(x, y):
    radius = np.sqrt(x * x + y * y)
    safe_radius = np.sqrt(radius * radius + LOOP_EPSILON * LOOP_EPSILON)
    angle = np.arctan2(y, x)
    log_radius = np.log(safe_radius)
    phase = log_radius * (math.tau / LOOP_LOG_PERIOD)
    wrapped_log_radius = (
        np.sin(phase) * LOOP_RADIAL_GAIN
        + np.sin(phase * 2.0 + 0.7) * LOOP_RADIAL_GAIN * LOOP_RADIAL_HARMONIC
    )
    twisted_angle = angle + wrapped_log_radius * LOOP_TWIST + np.sin(wrapped_log_radius * LOOP_JITTER_FREQ + 1.6180339) * LOOP_JITTER
    wrapped_radius = np.exp(wrapped_log_radius)
    radial_scale = radius / safe_radius
    return np.cos(twisted_angle) * wrapped_radius * radial_scale, np.sin(twisted_angle) * wrapped_radius * radial_scale


def animated_view(seed: FractalSeed, camera_elapsed: float, focus_a: tuple[float, float], focus_b: tuple[float, float]) -> tuple[float, float, float, float]:
    fractal_phase = camera_elapsed * seed.time_scale
    zoom_progress = camera_elapsed * ZOOM_RATE
    zoom_phase = zoom_progress % 1.0
    zoom_multiplier = 2.0 ** (zoom_phase * ZOOM_OCTAVES)
    focus_mix = smoothstep(zoom_phase)
    focus_x = focus_a[0] + (focus_b[0] - focus_a[0]) * focus_mix
    focus_y = focus_a[1] + (focus_b[1] - focus_a[1]) * focus_mix
    drift_x = math.sin(camera_elapsed * 0.19 + seed.hue_shift * math.tau) * DRIFT_STRENGTH / zoom_multiplier
    drift_y = math.cos(camera_elapsed * 0.23 - seed.hue_shift * (math.tau * 0.75)) * (DRIFT_STRENGTH * 0.74) / zoom_multiplier
    zoom = seed.zoom * zoom_multiplier
    return fractal_phase, zoom, focus_x + drift_x, focus_y + drift_y


def adaptive_max_iter(zoom: float) -> int:
    if not ADAPTIVE_DETAIL:
        return MAX_ITER
    effective_zoom = max(zoom, 1e-6)
    zoom_octaves = max(0.0, math.log2(effective_zoom))
    target = int(MIN_ITER + zoom_octaves * ITER_GROWTH_PER_OCTAVE)
    return max(MIN_ITER, min(MAX_ITER, target))


class FractalEngine:
    def __init__(self) -> None:
        self.seed = self.random_seed()
        self.frame_index = 0
        self.last_render_seconds = 0.0
        self.export_backend = self.default_export_backend()
        self._cupy_kernel = None
        self._focus_path: list[tuple[float, float]] = []
        self._loop_focus: tuple[float, float] | None = None
        self.seed = self.vetted_seed()
        self.reset_focus_path()

    def random_seed(self) -> FractalSeed:
        adjective = random.choice(
            ["Lattice", "Tidal", "Ferro", "Glass", "Velvet", "Echo", "Helix", "Oblique"]
        )
        noun = random.choice(
            ["Bloom", "Drift", "Cascade", "Spiral", "Weave", "Current", "Fissure", "Haze"]
        )
        return FractalSeed(
            name=f"{adjective} {noun}",
            polynomial_power=random.randint(2, 5),
            alt_power=random.randint(2, 4),
            blend=random.uniform(0.35, 0.85),
            swirl=random.uniform(0.08, 0.32),
            fold=random.uniform(0.12, 0.45),
            trig_scale=random.uniform(0.65, 2.1),
            drift=random.uniform(0.05, 0.24),
            warp=random.uniform(0.35, 1.7),
            hue_shift=random.random(),
            zoom=random.uniform(0.8, 1.8),
            pan_x=random.uniform(-0.45, 0.45),
            pan_y=random.uniform(-0.45, 0.45),
            time_scale=random.uniform(0.45, 1.35),
        )

    def reseed(self) -> None:
        self.seed = self.vetted_seed()
        self.frame_index = 0
        self.reset_focus_path()

    def vetted_seed(self) -> FractalSeed:
        best_seed = self.random_seed()
        best_score = -1e9

        for _ in range(SEED_VET_ATTEMPTS):
            candidate = self.random_seed()
            score = self.evaluate_seed(candidate)
            if score > best_score:
                best_score = score
                best_seed = candidate
            if score >= MIN_LOOP_SCORE:
                return candidate

        return best_seed

    def reset_focus_path(self) -> None:
        start_focus = self.find_bootstrap_focus()
        self.seed.pan_x, self.seed.pan_y = start_focus
        self._focus_path = [start_focus]
        for depth in range(FOCUS_DEPTH):
            self._focus_path.append(self.find_interesting_focus(depth))
        self._loop_focus = self.select_loop_focus()

    def select_loop_focus(self) -> tuple[float, float]:
        best_point = self._focus_path[0]
        best_score = -1e9
        for point in self._focus_path:
            score = self.score_focus_point(point[0], point[1], 0.01, 0.01)
            if score > best_score:
                best_score = score
                best_point = point
        return best_point

    def evaluate_seed(self, candidate: FractalSeed) -> float:
        original_seed = self.seed
        original_path = self._focus_path
        self.seed = candidate
        self._focus_path = []
        start_focus = self.find_bootstrap_focus()
        bootstrap_score = self.score_focus_point(start_focus[0], start_focus[1], 0.02, 0.02)

        if bootstrap_score < MIN_BOOTSTRAP_SCORE:
            self.seed = original_seed
            self._focus_path = original_path
            return bootstrap_score - 2.0

        self.seed.pan_x, self.seed.pan_y = start_focus
        self._focus_path = [start_focus]
        path_scores: list[float] = []
        path_motion = 0.0

        for depth in range(min(4, FOCUS_DEPTH)):
            next_focus = self.find_interesting_focus(depth)
            prev_focus = self._focus_path[-1]
            step = math.hypot(next_focus[0] - prev_focus[0], next_focus[1] - prev_focus[1])
            path_motion += step
            self._focus_path.append(next_focus)
            span = 0.01 / max(candidate.zoom * (2.0 ** depth), 1e-6)
            path_scores.append(self.score_focus_point(next_focus[0], next_focus[1], span, span))

        mean_path_score = sum(path_scores) / max(1, len(path_scores))
        repetition_penalty = 0.0 if path_motion >= MIN_LOOP_MOTION else 0.8
        score = bootstrap_score * 0.65 + mean_path_score * 0.9 + min(path_motion * 180.0, 0.7) - repetition_penalty

        self.seed = original_seed
        self._focus_path = original_path
        return score

    def find_bootstrap_focus(self) -> tuple[float, float]:
        seed = self.seed
        center_x, center_y = seed.pan_x, seed.pan_y
        span_x = 2.8 / max(seed.zoom, 1e-6)
        span_y = 2.1 / max(seed.zoom, 1e-6)
        best_point = (center_x, center_y)
        best_score = -1e9

        for gy in range(BOOTSTRAP_GRID):
            fy = (gy / (BOOTSTRAP_GRID - 1)) - 0.5
            for gx in range(BOOTSTRAP_GRID):
                fx = (gx / (BOOTSTRAP_GRID - 1)) - 0.5
                candidate_x = center_x + fx * span_x
                candidate_y = center_y + fy * span_y
                score = self.score_focus_point(
                    candidate_x,
                    candidate_y,
                    span_x * 0.08,
                    span_y * 0.08,
                )
                score -= (fx * fx + fy * fy) * 0.04
                if score > best_score:
                    best_score = score
                    best_point = (candidate_x, candidate_y)

        return best_point

    def focus_pair(self, camera_elapsed: float) -> tuple[tuple[float, float], tuple[float, float]]:
        if LOCK_LOOP_FOCUS and self._loop_focus is not None:
            return self._loop_focus, self._loop_focus
        loop_index = int(camera_elapsed * ZOOM_RATE) % max(1, len(self._focus_path) - 1)
        return self._focus_path[loop_index], self._focus_path[loop_index + 1]

    def find_interesting_focus(self, depth: int) -> tuple[float, float]:
        seed = self.seed
        center_x, center_y = self._focus_path[-1]
        search_zoom = seed.zoom * (2.0 ** (depth * (ZOOM_OCTAVES * 0.55)))
        span_x = 1.6 / max(search_zoom, 1e-6)
        span_y = 1.2 / max(search_zoom, 1e-6)

        best_x = center_x
        best_y = center_y
        best_score = -1e9

        for refine in range(FOCUS_REFINE_PASSES):
            current_span_x = span_x / (2.2**refine)
            current_span_y = span_y / (2.2**refine)
            local_best_x = best_x
            local_best_y = best_y
            local_best_score = best_score

            for gy in range(FOCUS_GRID):
                fy = (gy / (FOCUS_GRID - 1)) - 0.5
                for gx in range(FOCUS_GRID):
                    fx = (gx / (FOCUS_GRID - 1)) - 0.5
                    candidate_x = best_x + fx * current_span_x
                    candidate_y = best_y + fy * current_span_y
                    score = self.score_focus_point(
                        candidate_x,
                        candidate_y,
                        current_span_x * 0.16,
                        current_span_y * 0.16,
                    )
                    score -= (fx * fx + fy * fy) * 0.06
                    if score > local_best_score:
                        local_best_score = score
                        local_best_x = candidate_x
                        local_best_y = candidate_y

            best_x = local_best_x
            best_y = local_best_y
            best_score = local_best_score

        distance = math.hypot(best_x - center_x, best_y - center_y)
        minimum_step = min(span_x, span_y) * 0.02
        if distance < minimum_step:
            angle = depth * 2.399963229728653 + self.seed.hue_shift * math.tau
            best_x = center_x + math.cos(angle) * minimum_step
            best_y = center_y + math.sin(angle) * minimum_step

        return best_x, best_y

    def point_metrics(self, x: float, y: float) -> tuple[float, float, float]:
        seed = self.seed
        time_phase = 0.37 * seed.time_scale
        z_real = x * 0.35 + math.sin(y * seed.warp + time_phase) * 0.22
        z_imag = y * 0.35 + math.cos(x * seed.warp - time_phase * 0.7) * 0.22
        trap = 99.0
        orbit = 0.0
        mag2 = z_real * z_real + z_imag * z_imag
        iteration = 0

        while iteration < FOCUS_ITER:
            radius = math.sqrt(max(mag2, 1e-9))
            angle = math.atan2(z_imag, z_real)
            warped_real = x + math.sin(angle * seed.alt_power + time_phase) * 0.16 * seed.warp
            warped_imag = y + math.cos(radius * seed.trig_scale - time_phase) * 0.16 * seed.warp
            power = complex(z_real, z_imag) ** seed.polynomial_power
            primary_scale = 0.85 + 0.35 * math.sin(time_phase + radius * 0.6)
            secondary_real = math.sin(z_real * seed.trig_scale + time_phase) * seed.swirl
            secondary_imag = math.cos(z_imag * seed.trig_scale - time_phase * 0.8) * seed.swirl
            fold_scale = 1.0 / (1.0 + mag2)
            folded_real = math.sin((z_real * z_imag) * seed.fold + time_phase * 0.6) * fold_scale
            folded_imag = math.cos((z_real - z_imag) * seed.fold - time_phase * 0.45) * fold_scale
            z_real = (power.real + warped_real * primary_scale) * seed.blend + secondary_real + folded_real + warped_real * seed.drift
            z_imag = (power.imag + warped_imag * primary_scale) * seed.blend + secondary_imag + folded_imag + warped_imag * seed.drift
            mag2 = z_real * z_real + z_imag * z_imag
            trap = min(trap, abs(z_real * z_imag) + abs(math.sqrt(min(mag2, 1e12)) - 1.0))
            orbit += math.sin(angle * 3.0 + time_phase) * 0.5 + 0.5
            iteration += 1
            if mag2 > 64.0:
                break

        return iteration / FOCUS_ITER, math.exp(-trap * (4.0 + seed.fold * 3.5)), orbit / max(1, iteration)

    def score_focus_point(self, x: float, y: float, dx: float, dy: float) -> float:
        center_iter, center_trap, center_orbit = self.point_metrics(x, y)
        neighbors = (
            self.point_metrics(x + dx, y),
            self.point_metrics(x - dx, y),
            self.point_metrics(x, y + dy),
            self.point_metrics(x, y - dy),
        )

        mean_iter = sum(metric[0] for metric in neighbors) / len(neighbors)
        contrast = sum(abs(center_iter - metric[0]) for metric in neighbors) / len(neighbors)
        trap_contrast = sum(abs(center_trap - metric[1]) for metric in neighbors) / len(neighbors)
        orbit_mean = (center_orbit + sum(metric[2] for metric in neighbors)) / (len(neighbors) + 1)

        # Prefer boundary regions with local variation, not deep interior or immediately-empty exterior.
        boundary_band = 1.0 - abs(center_iter - 0.58)
        neighborhood_band = 1.0 - abs(mean_iter - 0.55)
        score = boundary_band * 1.3 + neighborhood_band * 0.8 + contrast * 2.4 + trap_contrast * 1.6 + orbit_mean * 0.2
        if center_iter < 0.08 or center_iter > 0.96:
            score -= 1.5
        if contrast < 0.015:
            score -= 0.8
        return score

    def available_export_backends(self) -> list[str]:
        backends = ["python"]
        if np is not None:
            backends.insert(0, "numpy")
        if cp is not None:
            backends.insert(0, "cupy")
        return backends

    def default_export_backend(self) -> str:
        requested = os.environ.get("FRACTAL_EXPORT_BACKEND", "").strip().lower()
        if requested in self.available_export_backends():
            return requested
        if cp is not None:
            return "cupy"
        if np is not None:
            return "numpy"
        return "python"

    def cycle_export_backend(self) -> str:
        backends = self.available_export_backends()
        index = (backends.index(self.export_backend) + 1) % len(backends)
        self.export_backend = backends[index]
        return self.export_backend

    def export_seed(self) -> Path:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{self.seed.name.lower().replace(' ', '_')}.json"
        path.write_text(json.dumps(asdict(self.seed), indent=2), encoding="utf-8")
        return path

    def load_seed(self, path: str | Path) -> Path:
        seed_path = Path(path)
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        self.seed = FractalSeed(**data)
        self.frame_index = 0
        self.reset_focus_path()
        return seed_path

    def export_frame(self, rgb: bytes, width: int, height: int, suffix: str | None = None) -> Path:
        OUTPUT_DIR.mkdir(exist_ok=True)
        name = self.seed.name.lower().replace(" ", "_")
        stem = f"{name}_{suffix}" if suffix else f"{name}_{self.frame_index:04d}"
        path = OUTPUT_DIR / f"{stem}.ppm"
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        path.write_bytes(header + rgb)
        return path

    def shader_uniforms(self, camera_elapsed: float, fractal_elapsed: float) -> dict[str, float | int | tuple[float, float]]:
        seed = self.seed
        _, zoom, pan_x, pan_y = animated_view(seed, camera_elapsed, *self.focus_pair(camera_elapsed))
        max_iter = adaptive_max_iter(zoom)
        return {
            "max_iter": max_iter,
            "polynomial_power": seed.polynomial_power,
            "alt_power": seed.alt_power,
            "blend": seed.blend,
            "swirl": seed.swirl,
            "fold": seed.fold,
            "trig_scale": seed.trig_scale,
            "drift": seed.drift,
            "warp": seed.warp,
            "hue_shift": seed.hue_shift,
            "zoom_base": seed.zoom,
            "pan": (pan_x, pan_y),
            "time_scale": seed.time_scale,
            "fractal_elapsed": fractal_elapsed,
        }

    def render_export(self, width: int, height: int, camera_elapsed: float, fractal_elapsed: float | None = None) -> bytes:
        if fractal_elapsed is None:
            fractal_elapsed = camera_elapsed
        if self.export_backend == "cupy" and cp is not None:
            return self.render_cupy_raw(width, height, camera_elapsed, fractal_elapsed)
        if self.export_backend == "numpy" and np is not None:
            return self.render_numpy(width, height, camera_elapsed, fractal_elapsed)
        return self.render_python(width, height, camera_elapsed, fractal_elapsed)

    def _get_cupy_kernel(self):
        if self._cupy_kernel is None and cp is not None:
            self._cupy_kernel = cp.RawKernel(CUPY_KERNEL_SOURCE, "render_fractal")
        return self._cupy_kernel

    def render_cupy_raw(self, width: int, height: int, camera_elapsed: float, fractal_elapsed: float | None = None) -> bytes:
        start = time.perf_counter()
        if fractal_elapsed is None:
            fractal_elapsed = camera_elapsed
        kernel = self._get_cupy_kernel()
        output = cp.empty(width * height * 3, dtype=cp.uint8)
        threads = 256
        blocks = (width * height + threads - 1) // threads
        seed = self.seed
        _, zoom, pan_x, pan_y = animated_view(seed, camera_elapsed, *self.focus_pair(camera_elapsed))
        max_iter = adaptive_max_iter(zoom)
        kernel(
            (blocks,),
            (threads,),
            (
                output,
                width,
                height,
                float(fractal_elapsed),
                max_iter,
                seed.polynomial_power,
                seed.alt_power,
                float(seed.blend),
                float(seed.swirl),
                float(seed.fold),
                float(seed.trig_scale),
                float(seed.drift),
                float(seed.warp),
                float(seed.hue_shift),
                float(seed.zoom),
                float(pan_x),
                float(pan_y),
                float(seed.time_scale),
            ),
        )
        cp.cuda.Stream.null.synchronize()
        rgb = cp.asnumpy(output).tobytes()
        self.frame_index += 1
        self.last_render_seconds = time.perf_counter() - start
        return rgb

    def render_numpy(self, width: int, height: int, camera_elapsed: float, fractal_elapsed: float | None = None) -> bytes:
        start = time.perf_counter()
        seed = self.seed
        if fractal_elapsed is None:
            fractal_elapsed = camera_elapsed
        _, zoom, pan_x, pan_y = animated_view(seed, camera_elapsed, *self.focus_pair(camera_elapsed))
        max_iter = adaptive_max_iter(zoom)
        time_phase = fractal_elapsed * seed.time_scale
        x = np.linspace(-2.0, 2.0, width, endpoint=False, dtype=np.float32)
        y = np.linspace(-1.5, 1.5, height, endpoint=False, dtype=np.float32)
        local_x = x[None, :] / zoom
        local_y = y[:, None] / zoom
        mapped_x, mapped_y = loop_map_arrays(local_x, local_y)
        nx = mapped_x + pan_x
        ny = mapped_y + pan_y
        c_real = np.broadcast_to(nx, (height, width))
        c_imag = np.broadcast_to(ny, (height, width))
        z_real = c_real * 0.35 + np.sin(c_imag * seed.warp + time_phase) * 0.22
        z_imag = c_imag * 0.35 + np.cos(c_real * seed.warp - time_phase * 0.7) * 0.22
        orbit = np.zeros((height, width), dtype=np.float32)
        trap = np.full((height, width), 99.0, dtype=np.float32)
        smooth = np.full((height, width), float(max_iter), dtype=np.float32)
        active = np.ones((height, width), dtype=bool)

        for iteration in range(max_iter):
            if not np.any(active):
                break
            mag2 = z_real * z_real + z_imag * z_imag
            radius = np.sqrt(np.maximum(mag2, 1e-9))
            angle = np.arctan2(z_imag, z_real)
            warped_real = c_real + np.sin(angle * seed.alt_power + time_phase) * 0.16 * seed.warp
            warped_imag = c_imag + np.cos(radius * seed.trig_scale - time_phase) * 0.16 * seed.warp

            if seed.polynomial_power == 2:
                power_real = z_real * z_real - z_imag * z_imag
                power_imag = 2.0 * z_real * z_imag
            elif seed.polynomial_power == 3:
                r2 = z_real * z_real
                i2 = z_imag * z_imag
                power_real = z_real * (r2 - 3.0 * i2)
                power_imag = z_imag * (3.0 * r2 - i2)
            else:
                power_complex = (z_real + 1j * z_imag) ** seed.polynomial_power
                power_real = power_complex.real.astype(np.float32)
                power_imag = power_complex.imag.astype(np.float32)

            primary_scale = 0.85 + 0.35 * np.sin(time_phase + radius * 0.6)
            secondary_real = np.sin(z_real * seed.trig_scale + time_phase) * seed.swirl
            secondary_imag = np.cos(z_imag * seed.trig_scale - time_phase * 0.8) * seed.swirl
            fold_scale = 1.0 / (1.0 + mag2)
            folded_real = np.sin((z_real * z_imag) * seed.fold + time_phase * 0.6) * fold_scale
            folded_imag = np.cos((z_real - z_imag) * seed.fold - time_phase * 0.45) * fold_scale

            z_real = np.where(
                active,
                (power_real + warped_real * primary_scale) * seed.blend + secondary_real + folded_real + warped_real * seed.drift,
                z_real,
            )
            z_imag = np.where(
                active,
                (power_imag + warped_imag * primary_scale) * seed.blend + secondary_imag + folded_imag + warped_imag * seed.drift,
                z_imag,
            )
            next_mag2 = z_real * z_real + z_imag * z_imag
            escaped_now = active & (next_mag2 > 64.0)
            safe_mag2 = np.clip(next_mag2, 1.0001, 1e20)
            smooth[escaped_now] = iteration + 1.0 - np.log2(np.maximum(np.log2(safe_mag2[escaped_now]), 1e-6))
            trap = np.minimum(trap, np.abs(z_real * z_imag) + np.abs(np.sqrt(np.minimum(next_mag2, 1e12)) - 1.0))
            orbit += active * (np.sin(angle * 3.0 + time_phase) * 0.5 + 0.5)
            active &= ~escaped_now

        density = smooth / max_iter
        trap_glow = np.exp(-trap * (4.0 + seed.fold * 3.5))
        hue = (seed.hue_shift + density * 0.45 + trap_glow * 0.28 + orbit * 0.04) % 1.0
        saturation = np.clip(0.45 + trap_glow * 0.75 + 0.15 * np.sin(time_phase + c_real), 0.0, 1.0)
        value = np.clip(density * 0.85 + trap_glow * 0.65 + orbit * 0.03, 0.0, 1.0)
        value = np.where(active, value * 0.16, value)
        saturation = np.where(active, saturation * 0.25, saturation)

        rgb = np.empty((height, width, 3), dtype=np.uint8)
        for py in range(height):
            for px in range(width):
                rgb[py, px] = hsv_to_rgb_int(float(hue[py, px]), float(saturation[py, px]), float(value[py, px]))
        self.frame_index += 1
        self.last_render_seconds = time.perf_counter() - start
        return rgb.tobytes()

    def render_python(self, width: int, height: int, camera_elapsed: float, fractal_elapsed: float | None = None) -> bytes:
        start = time.perf_counter()
        seed = self.seed
        if fractal_elapsed is None:
            fractal_elapsed = camera_elapsed
        _, zoom, pan_x, pan_y = animated_view(seed, camera_elapsed, *self.focus_pair(camera_elapsed))
        max_iter = adaptive_max_iter(zoom)
        time_phase = fractal_elapsed * seed.time_scale
        inverse_zoom = 1.0 / zoom
        buffer = bytearray(width * height * 3)
        index = 0

        for py in range(height):
            local_y = ((py / height) - 0.5) * (3.0 * inverse_zoom)
            for px in range(width):
                local_x = ((px / width) - 0.5) * (4.0 * inverse_zoom)
                mapped_x, mapped_y = loop_map_point(local_x, local_y)
                nx = mapped_x + pan_x
                ny = mapped_y + pan_y
                z_real = nx * 0.35 + math.sin(ny * seed.warp + time_phase) * 0.22
                z_imag = ny * 0.35 + math.cos(nx * seed.warp - time_phase * 0.7) * 0.22
                orbit = 0.0
                trap = 99.0
                mag2 = z_real * z_real + z_imag * z_imag
                iteration = 0

                while iteration < max_iter:
                    radius = math.sqrt(max(mag2, 1e-9))
                    angle = math.atan2(z_imag, z_real)
                    warped_real = nx + math.sin(angle * seed.alt_power + time_phase) * 0.16 * seed.warp
                    warped_imag = ny + math.cos(radius * seed.trig_scale - time_phase) * 0.16 * seed.warp
                    power = complex(z_real, z_imag) ** seed.polynomial_power
                    primary_scale = 0.85 + 0.35 * math.sin(time_phase + radius * 0.6)
                    secondary_real = math.sin(z_real * seed.trig_scale + time_phase) * seed.swirl
                    secondary_imag = math.cos(z_imag * seed.trig_scale - time_phase * 0.8) * seed.swirl
                    fold_scale = 1.0 / (1.0 + mag2)
                    folded_real = math.sin((z_real * z_imag) * seed.fold + time_phase * 0.6) * fold_scale
                    folded_imag = math.cos((z_real - z_imag) * seed.fold - time_phase * 0.45) * fold_scale
                    z_real = (power.real + warped_real * primary_scale) * seed.blend + secondary_real + folded_real + warped_real * seed.drift
                    z_imag = (power.imag + warped_imag * primary_scale) * seed.blend + secondary_imag + folded_imag + warped_imag * seed.drift
                    mag2 = z_real * z_real + z_imag * z_imag
                    orbit += math.sin(angle * 3.0 + time_phase) * 0.5 + 0.5
                    trap = min(trap, abs(z_real * z_imag) + abs(math.sqrt(min(mag2, 1e12)) - 1.0))
                    iteration += 1
                    if mag2 > 64.0:
                        break

                smooth = float(iteration)
                if mag2 > 1.0 and mag2 < 1e20:
                    smooth = iteration + 1.0 - math.log2(max(math.log2(mag2), 1e-6))
                density = smooth / max_iter
                trap_glow = math.exp(-trap * (4.0 + seed.fold * 3.5))
                orbit_mix = orbit / max(iteration, 1)
                hue = (seed.hue_shift + density * 0.45 + trap_glow * 0.28 + orbit_mix * 0.18) % 1.0
                saturation = max(0.0, min(1.0, 0.45 + trap_glow * 0.75 + 0.15 * math.sin(time_phase + nx)))
                value = max(0.0, min(1.0, density * 0.85 + trap_glow * 0.65 + orbit_mix * 0.25))
                if mag2 <= 64.0:
                    value *= 0.16
                    saturation *= 0.25
                red, green, blue = hsv_to_rgb_int(hue, saturation, value)
                buffer[index] = red
                buffer[index + 1] = green
                buffer[index + 2] = blue
                index += 3

        self.frame_index += 1
        self.last_render_seconds = time.perf_counter() - start
        return bytes(buffer)


class ShaderWindow(pyglet.window.Window):
    def __init__(self, engine: FractalEngine, smoke_test: bool = False, visible: bool = True, target_fps: float = TARGET_FPS) -> None:
        config = pyglet.gl.Config(double_buffer=True, major_version=3, minor_version=3)
        super().__init__(
            width=WIDTH,
            height=HEIGHT,
            caption="Synthetic Fractals",
            resizable=True,
            config=config,
            vsync=False,
            visible=visible and not smoke_test,
        )
        self.engine = engine
        self.running = True
        self.fractal_running = ANIMATE_FRACTAL
        self.start_time = time.perf_counter()
        self.camera_start_time = self.start_time
        self.frozen_elapsed = 0.0
        self.fractal_frozen_elapsed = 0.0
        self.last_draw_seconds = 0.0
        self.windowed_size = (WIDTH, HEIGHT)
        self.audio_track: AudioReactiveTrack | None = None
        self.audio_player = None
        self.audio_level = 0.0
        self.audio_beat = 0.0
        self.audio_groove = 0.0
        self.music_sensitivity = MUSIC_SENSITIVITY
        self.recording_dir: Path | None = None
        self.recording_frame_limit: int | None = None
        self.recording_frame_index = 0
        self.recording_started_at = 0.0
        self._recording_queue: Queue[tuple[Path, bytes, int, int] | None] | None = None
        self._recording_thread: Thread | None = None
        self.fixed_timestep: float | None = None
        self.fixed_camera_elapsed = 0.0
        self.fixed_fractal_elapsed = 0.0
        self.ctx = moderngl.create_context()
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        vertices = self.ctx.buffer(struct.pack("8f", -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
        self.vao = self.ctx.simple_vertex_array(self.program, vertices, "in_pos")
        self.debug_enabled = True
        self.debug_label = pyglet.text.Label(
            "",
            x=16,
            y=self.height - 16,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            width=max(320, self.width - 32),
            color=(230, 240, 245, 255),
        )
        pyglet.clock.schedule_interval(self.tick, 1.0 / target_fps)
        if smoke_test:
            pyglet.clock.schedule_once(lambda _dt: self.close(), 0.2)

    def _ensure_recording_writer(self) -> None:
        if self._recording_thread is not None:
            return
        self._recording_queue = Queue(maxsize=RECORD_QUEUE_MAX)

        def worker() -> None:
            assert self._recording_queue is not None
            while True:
                item = self._recording_queue.get()
                if item is None:
                    self._recording_queue.task_done()
                    break
                path, rgb, width, height = item
                write_recording_frame(path, rgb, width, height)
                self._recording_queue.task_done()

        self._recording_thread = Thread(target=worker, name='recording-writer', daemon=True)
        self._recording_thread.start()

    def _stop_recording_writer(self) -> None:
        if self._recording_thread is None or self._recording_queue is None:
            return
        self._recording_queue.put(None)
        self._recording_queue.join()
        self._recording_thread.join()
        self._recording_thread = None
        self._recording_queue = None

    def elapsed(self) -> float:
        return self.camera_elapsed()

    def camera_elapsed(self) -> float:
        if self.fixed_timestep is not None:
            return self.fixed_camera_elapsed
        if self.running:
            return time.perf_counter() - self.camera_start_time
        return self.frozen_elapsed

    def fractal_elapsed(self) -> float:
        if self.fixed_timestep is not None:
            return self.fixed_fractal_elapsed
        if self.fractal_running:
            return time.perf_counter() - self.start_time
        return self.fractal_frozen_elapsed

    def update_uniforms(self) -> None:
        camera_elapsed = self.camera_elapsed()
        fractal_elapsed = self.fractal_elapsed()
        uniforms = self.engine.shader_uniforms(camera_elapsed, fractal_elapsed)
        _level, kick = self.current_audio_features()
        if self.audio_track is not None:
            sensitivity = self.music_sensitivity
            groove_drive = min(1.0, self.audio_groove * sensitivity)
            accent_drive = kick * sensitivity * 0.28
            uniforms["hue_shift"] = (float(uniforms["hue_shift"]) + (groove_drive * 0.120 + accent_drive * 0.050) * 15) % 1.0
        self.program["elapsed"] = camera_elapsed
        for name, value in uniforms.items():
            if name in self.program:
                self.program[name] = value

    def update_debug_overlay(self) -> None:
        camera_elapsed = self.camera_elapsed()
        focus_a, focus_b = self.engine.focus_pair(camera_elapsed)
        _, zoom, pan_x, pan_y = animated_view(self.engine.seed, camera_elapsed, focus_a, focus_b)
        zoom_phase = (camera_elapsed * ZOOM_RATE) % 1.0
        zoom_multiplier = 2.0 ** (zoom_phase * ZOOM_OCTAVES)
        current_iter = adaptive_max_iter(zoom)
        fps = 1.0 / self.last_draw_seconds if self.last_draw_seconds > 1e-9 else 0.0
        music_text = "music off"
        if self.audio_track is not None:
            music_text = f"music {self.audio_track.path.stem}  level {self.audio_level:.2f}  kick {self.audio_beat:.2f}  groove {self.audio_groove:.2f}  sens {self.music_sensitivity:.2f}"
        self.debug_label.text = (
            f"fps {fps:,.1f}  phase {zoom_phase:.4f}  zoom x{zoom_multiplier:,.1f}  iter {current_iter}\n"
            f"{music_text}\n"
            f"focus A ({focus_a[0]:.6f}, {focus_a[1]:.6f})\n"
            f"focus B ({focus_b[0]:.6f}, {focus_b[1]:.6f})\n"
            f"pan ({pan_x:.6f}, {pan_y:.6f})"
        )
        self.debug_label.y = self.height - 16
        self.debug_label.width = max(320, self.width - 32)

    def tick(self, _dt: float) -> None:
        if getattr(self, "context", None) is None:
            return
        self.switch_to()
        self.dispatch_event("on_draw")
        self.flip()

    def on_draw(self) -> None:
        frame_start = time.perf_counter()
        self.switch_to()
        self.ctx.clear()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.update_uniforms()
        self.vao.render(moderngl.TRIANGLE_STRIP)
        if self.debug_enabled:
            self.update_debug_overlay()
            self.debug_label.draw()
        if self.recording_dir is not None:
            self._ensure_recording_writer()
            raw = self.ctx.screen.read(viewport=(0, 0, self.width, self.height), components=3, alignment=1)
            stride = self.width * 3
            rows = [raw[index : index + stride] for index in range(0, len(raw), stride)]
            rgb = b"".join(reversed(rows))
            assert self._recording_queue is not None
            self._recording_queue.put((self.recording_dir / f"frame_{self.recording_frame_index:06d}.jpg", rgb, self.width, self.height))
            self.recording_frame_index += 1
            if self.recording_frame_limit is not None:
                elapsed = max(1e-6, time.perf_counter() - self.recording_started_at)
                fps_done = self.recording_frame_index / elapsed
                remaining = max(0, self.recording_frame_limit - self.recording_frame_index)
                eta_seconds = remaining / max(fps_done, 1e-6)
                eta_total = int(max(0.0, eta_seconds))
                eta_minutes, eta_secs = divmod(eta_total, 60)
                pending = self._recording_queue.qsize()
                print(
                    f"\rrecording frame {self.recording_frame_index}/{self.recording_frame_limit} | eta {eta_minutes:02d}:{eta_secs:02d} | queued {pending}",
                    end="",
                    flush=True,
                )
            if self.recording_frame_limit is not None and self.recording_frame_index >= self.recording_frame_limit:
                self._stop_recording_writer()
                print()
                print(f"saved recording frames to {self.recording_dir}")
                pyglet.clock.unschedule(self.tick)
                pyglet.clock.schedule_once(lambda _dt: pyglet.app.exit(), 0.0)
                return
        if self.fixed_timestep is not None:
            if self.running:
                self.fixed_camera_elapsed += self.fixed_timestep
            if self.fractal_running:
                self.fixed_fractal_elapsed += self.fixed_timestep
        self.engine.frame_index += 1
        self.last_draw_seconds = time.perf_counter() - frame_start
        self.set_caption(
            f"{self.engine.seed.name} | shader {self.last_draw_seconds * 1000:.1f} ms | "
            f"export {self.engine.export_backend} | "
            f"{'zoom' if self.running else 'paused'} | "
            f"{'fractal' if self.fractal_running else 'fractal-frozen'}"
        )

    def on_key_press(self, symbol: int, _modifiers: int) -> None:
        if symbol == key.R:
            self.engine.reseed()
        elif symbol == key.M:
            path = self.choose_music_file()
            if path is not None:
                try:
                    loaded = self.load_music(path)
                except (OSError, ValueError) as exc:
                    print(f"failed to load music: {exc}")
                else:
                    print(f"loaded music from {loaded}")
        elif symbol == key.UP:
            self.music_sensitivity = min(5.0, self.music_sensitivity + 0.1)
            print(f"music sensitivity -> {self.music_sensitivity:.2f}")
        elif symbol == key.DOWN:
            self.music_sensitivity = max(0.0, self.music_sensitivity - 0.1)
            print(f"music sensitivity -> {self.music_sensitivity:.2f}")
        elif symbol == key.L:
            path = self.choose_seed_file()
            if path is not None:
                try:
                    loaded = self.engine.load_seed(path)
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    print(f"failed to load seed: {exc}")
                else:
                    print(f"loaded seed from {loaded}")
        elif symbol == key.SPACE:
            self.toggle()
        elif symbol == key.A:
            self.toggle_fractal_animation()
        elif symbol == key.S:
            path = self.engine.export_seed()
            print(f"saved seed to {path}")
        elif symbol == key.P:
            path = self.save_frame()
            print(f"saved frame to {path}")
        elif symbol == key.B:
            backend = self.engine.cycle_export_backend()
            print(f"export backend -> {backend}")
        elif symbol == key.D:
            self.debug_enabled = not self.debug_enabled
        elif symbol == key.F:
            self.toggle_fullscreen()
        elif symbol == key.ESCAPE:
            if self.fullscreen:
                self.toggle_fullscreen()
            else:
                self.close()

    def on_close(self) -> None:
        self._stop_recording_writer()
        super().on_close()

    def toggle(self) -> None:
        if self.running:
            self.frozen_elapsed = time.perf_counter() - self.camera_start_time
        else:
            self.camera_start_time = time.perf_counter() - self.frozen_elapsed
        self.running = not self.running

    def toggle_fractal_animation(self) -> None:
        if self.fractal_running:
            self.fractal_frozen_elapsed = time.perf_counter() - self.start_time
        else:
            self.start_time = time.perf_counter() - self.fractal_frozen_elapsed
        self.fractal_running = not self.fractal_running

    def toggle_fullscreen(self) -> None:
        if self.fullscreen:
            self.set_fullscreen(False, width=self.windowed_size[0], height=self.windowed_size[1])
        else:
            self.windowed_size = (self.width, self.height)
            self.set_fullscreen(True)

    def current_audio_features(self) -> tuple[float, float]:
        if self.audio_track is None or self.audio_player is None:
            self.audio_level = 0.0
            self.audio_beat = 0.0
            self.audio_groove = 0.0
            return 0.0, 0.0
        if self.fixed_timestep is not None and self.recording_dir is not None:
            playback_time = self.camera_elapsed()
        else:
            playback_time = getattr(self.audio_player, "time", 0.0) or 0.0
        level, beat, groove = self.audio_track.features_at(min(playback_time, self.audio_track.duration))
        self.audio_level = level
        self.audio_beat = beat
        self.audio_groove = groove
        return level, beat

    def load_music(self, path: str | Path) -> Path:
        if pyglet is None:
            raise RuntimeError("pyglet is required for music playback")
        track_path = Path(path)
        audio_track = AudioReactiveTrack.from_wav(track_path)
        source = pyglet.media.load(str(track_path), streaming=False)
        if self.audio_player is not None:
            self.audio_player.pause()
        player = pyglet.media.Player()
        player.queue(source)
        player.play()
        self.audio_track = audio_track
        self.audio_player = player
        self.audio_level = 0.0
        self.audio_beat = 0.0
        self.audio_groove = 0.0
        return track_path

    def choose_seed_file(self) -> Path | None:
        if tk is None or filedialog is None:
            print("tkinter is not available; cannot open file chooser")
            return None
        picker = tk.Tk()
        picker.withdraw()
        picker.attributes("-topmost", True)
        try:
            selected = filedialog.askopenfilename(
                title="Load Fractal Seed",
                initialdir=str(OUTPUT_DIR),
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
        finally:
            picker.destroy()
        return Path(selected) if selected else None

    def choose_music_file(self) -> Path | None:
        if tk is None or filedialog is None:
            print("tkinter is not available; cannot open file chooser")
            return None
        picker = tk.Tk()
        picker.withdraw()
        picker.attributes("-topmost", True)
        try:
            selected = filedialog.askopenfilename(
                title="Load Music",
                initialdir=str(Path.cwd()),
                filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            )
        finally:
            picker.destroy()
        return Path(selected) if selected else None

    def save_frame(self) -> Path:
        camera_elapsed = self.camera_elapsed()
        fractal_elapsed = self.fractal_elapsed()
        if self.engine.export_backend == "cupy":
            rgb = self.engine.render_cupy_raw(RENDER_WIDTH, RENDER_HEIGHT, camera_elapsed, fractal_elapsed)
            return self.engine.export_frame(rgb, RENDER_WIDTH, RENDER_HEIGHT, suffix="cupy")
        if self.engine.export_backend == "numpy":
            rgb = self.engine.render_numpy(RENDER_WIDTH, RENDER_HEIGHT, camera_elapsed, fractal_elapsed)
            return self.engine.export_frame(rgb, RENDER_WIDTH, RENDER_HEIGHT, suffix="numpy")
        if self.engine.export_backend == "python":
            rgb = self.engine.render_python(RENDER_WIDTH, RENDER_HEIGHT, camera_elapsed, fractal_elapsed)
            return self.engine.export_frame(rgb, RENDER_WIDTH, RENDER_HEIGHT, suffix="python")

        raw = self.ctx.screen.read(viewport=(0, 0, self.width, self.height), components=3, alignment=1)
        stride = self.width * 3
        rows = [raw[index : index + stride] for index in range(0, len(raw), stride)]
        rgb = b"".join(reversed(rows))
        return self.engine.export_frame(rgb, self.width, self.height, suffix="shader")


def benchmark(engine: FractalEngine, width: int, height: int, elapsed: float, rounds: int) -> None:
    results: list[tuple[str, list[float]]] = []
    for backend in engine.available_export_backends():
        times: list[float] = []
        engine.export_backend = backend
        for _ in range(rounds):
            engine.render_export(width, height, elapsed)
            times.append(engine.last_render_seconds)
        results.append((backend, times))

    print(f"benchmark {width}x{height}")
    for backend, times in results:
        warm = times[1:] if len(times) > 1 else times
        average = sum(warm) / len(warm)
        print(f"{backend:>6} all={[round(t, 4) for t in times]} avg_ex_first={average:.4f}s")


def write_recording_frame(path: Path, rgb: bytes, width: int, height: int) -> None:
    if Image is None:
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        path.with_suffix(".ppm").write_bytes(header + rgb)
        return
    image = Image.frombytes("RGB", (width, height), rgb)
    image.save(path, format="JPEG", quality=RECORD_JPEG_QUALITY, subsampling=RECORD_JPEG_SUBSAMPLING, optimize=False)


def prepare_fullscreen_start(window: ShaderWindow) -> None:
    was_running = window.running
    was_fractal_running = window.fractal_running
    window.running = False
    window.fractal_running = False
    window.frozen_elapsed = 0.0
    window.fractal_frozen_elapsed = 0.0
    window.on_draw()
    window.flip()
    time.sleep(FULLSCREEN_START_DELAY)
    now = time.perf_counter()
    window.camera_start_time = now
    window.start_time = now
    window.frozen_elapsed = 0.0
    window.fractal_frozen_elapsed = 0.0
    window.running = was_running
    window.fractal_running = was_fractal_running


def record_sequence(engine: FractalEngine, width: int, height: int, duration_seconds: float, audio_path: str | None, fullscreen: bool = False, hide_debug: bool = False) -> None:
    if moderngl is None or pyglet is None:
        raise RuntimeError("pyglet and moderngl are required for recording")
    if duration_seconds <= 0.0:
        raise ValueError("record duration must be greater than 0")

    RECORDINGS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    recording_dir = RECORDINGS_DIR / f"{engine.seed.name.lower().replace(' ', '_')}_{stamp}"
    recording_dir.mkdir(exist_ok=True)

    frame_rate = 60.0
    frame_count = max(1, int(math.ceil(duration_seconds * frame_rate)))

    window = ShaderWindow(engine, visible=True, target_fps=frame_rate)
    window.fixed_timestep = 1.0 / frame_rate
    if fullscreen:
        window.toggle_fullscreen()
        prepare_fullscreen_start(window)
    else:
        window.set_size(width, height)
    window.debug_enabled = not hide_debug
    window.recording_dir = recording_dir
    window.recording_frame_limit = frame_count
    window.recording_started_at = time.perf_counter()
    if audio_path is not None:
        window.load_music(audio_path)
    pyglet.app.run()
    if getattr(window, "context", None) is not None:
        window.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic fractal generator")
    parser.add_argument("--benchmark", action="store_true", help="benchmark export backends and exit")
    parser.add_argument("--width", type=int, default=RENDER_WIDTH, help="benchmark/export width")
    parser.add_argument("--height", type=int, default=RENDER_HEIGHT, help="benchmark/export height")
    parser.add_argument("--rounds", type=int, default=4, help="benchmark rounds")
    parser.add_argument("--seed", type=str, help="path to a saved seed JSON file")
    parser.add_argument("--audio", type=str, help="path to a WAV file for music-reactive playback/recording")
    parser.add_argument(
        "--record",
        nargs="?",
        const=-1.0,
        type=float,
        help="render an offline 60 FPS frame sequence; optional value is duration in seconds, defaults to audio duration if --audio is set",
    )
    parser.add_argument("--fullscreen", action="store_true", help="start the window in fullscreen")
    parser.add_argument("--hide-debug", action="store_true", help="start with the debug overlay hidden")
    parser.add_argument("--smoke-test", action="store_true", help="open the shader window briefly, then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = FractalEngine()
    if args.seed is not None:
        engine.load_seed(args.seed)

    if args.benchmark:
        benchmark(engine, args.width, args.height, 0.2, args.rounds)
        return

    if args.record is not None:
        duration_seconds = args.record
        if duration_seconds == -1.0:
            if args.audio is None:
                raise RuntimeError("--record without a duration requires --audio")
            duration_seconds = AudioReactiveTrack.from_wav(args.audio).duration
        record_sequence(engine, args.width, args.height, duration_seconds, args.audio, fullscreen=args.fullscreen, hide_debug=args.hide_debug)
        return

    if moderngl is None or pyglet is None:
        raise RuntimeError("pyglet and moderngl are required for the live window")

    window = ShaderWindow(engine, smoke_test=args.smoke_test)
    window.debug_enabled = not args.hide_debug
    if args.fullscreen:
        window.toggle_fullscreen()
        prepare_fullscreen_start(window)
    if args.audio is not None:
        window.load_music(args.audio)
    pyglet.app.run()


if __name__ == "__main__":
    main()
