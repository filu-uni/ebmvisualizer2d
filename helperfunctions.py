from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
#from collections import Counter
import os
from pathlib import Path
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PySide6.QtCore import Signal, QThread
import pyarrow
import polars as pl
from natsort import natsorted

from OpenGL.GL import (
    glClearColor, glClear, GL_COLOR_BUFFER_BIT,
    glUseProgram, glUniformMatrix4fv, glUniform1f, glUniform1i,
    glBindVertexArray, glGenVertexArrays,
    glBufferData, glGenBuffers, glBindBuffer, GL_ARRAY_BUFFER, GL_STATIC_DRAW,
    glVertexAttribPointer, glEnableVertexAttribArray,
    glDrawArrays, GL_POINTS, GL_FALSE, GL_TRUE,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv,
    GL_LINK_STATUS, glGetProgramInfoLog, glDeleteShader,
    glGetUniformLocation, glViewport,
    glGenTextures, glBindTexture, glTexImage1D, glTexParameteri,
    GL_TEXTURE_1D, GL_RGBA32F, GL_RGBA, GL_FLOAT, GL_LINEAR,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
    glCreateShader,glShaderSource,glCompileShader,
    glGetShaderiv,GL_COMPILE_STATUS,glGetShaderInfoLog
)

def get_wav_files(directory):
    base_path = Path(str(directory).strip())

    if not base_path.is_dir():
        print(f"Warning: {base_path} is not a valid directory!")
        return []
    return natsorted(list(base_path.glob("*.wav")))

def get_arrow_files(directory):
    """Returns a list of all .arrow files in the specified directory."""
    base_path = Path(str(directory).strip())

    if not base_path.is_dir():
        print(f"Warning: {base_path} is not a valid directory!")
        return []
    return natsorted(list(base_path.glob("*.arrow")))


def create_histogram_from_arrow_folder(folder_path, ch="channel_1", bins=100):
    folder = Path(folder_path)
    pattern = str(folder / "*.arrow")
    
    hist_df = (
    pl.scan_ipc(pattern)
    .select(ch)
    .group_by(ch)
    .len()
    .sort(ch) 
    .collect(streaming=True)
    )
    return hist_df

def union_sum_scaled_fast(a, b, scale):
    coords = np.vstack((a[:, :2], b[:, :2]))
    values = np.concatenate((a[:, 2], b[:, 2])) / scale

    order = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[order]
    values = values[order]

    unique_mask = np.any(np.diff(coords, axis=0), axis=1)
    idx = np.concatenate(([0], np.nonzero(unique_mask)[0] + 1))

    unique_coords = coords[idx]
    summed_values = np.add.reduceat(values, idx)

    return np.column_stack((unique_coords, summed_values))


def create_arrow_from_wav(file_path, number, out_folder="arrow_files"):
    out_dir = Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Layer_{number}.arrow"

    samplerate, data = wavfile.read(file_path)
    
    df = pl.DataFrame({
        "x": data[:, -4].astype(np.float32),
        "y": data[:, -3].astype(np.float32),
        "channel_1": data[:, 0].astype(np.float32),
        "channel_2": data[:, 1].astype(np.float32),
        "channel_3": data[:, 2].astype(np.float32),
        "channel_4": data[:, 3].astype(np.float32)
    })

    df.write_ipc(out_file)
    del data
    
    del df
    
    print(f"Exported to {out_file}")
    return out_file
def get_df_from_arrow(file, ch="channel_1", nth=1):
    file_path = Path(file).absolute()
    
    ldf = pl.scan_ipc(file_path)
    
    if nth > 1:
        ldf = ldf.gather_every(nth)
    
    
    ldf = ldf.select(["x", "y", ch])
    
    return ldf

def normalize_data(ldf, ch):
    
    ranges = ldf.select([
        pl.col("x").min().alias("xmin"),
        pl.col("x").max().alias("xmax"),
        pl.col("y").min().alias("ymin"),
        pl.col("y").max().alias("ymax"),
    ]).collect()

    xmin, xmax = ranges["xmin"][0], ranges["xmax"][0]
    ymin, ymax = ranges["ymin"][0], ranges["ymax"][0]
    
    
    ldf = ldf.with_columns([
        (2.0 * (pl.col("x") - xmin) / (xmax - xmin) - 1.0).alias("x_norm"),
        (2.0 * (pl.col("y") - ymin) / (ymax - ymin) - 1.0).alias("y_norm")
    ])
    
    ldf = ldf.select(["x_norm", "y_norm", ch])

    return ldf

class DataWorker(QThread):
    data_ready = Signal(object) 

    def __init__(self, nth, ch, file):
        super().__init__()
        self.nth = nth
        self.ch = ch
        self.file = file

    def run(self):
        ldf = get_df_from_arrow(self.file, self.ch, self.nth)
        ldf = normalize_data(ldf, self.ch)

        df = ldf.collect()

        arr = df.to_numpy()

        arr = np.ascontiguousarray(arr)
        self.data_ready.emit(arr) 
        self.finished.emit()


class WavHandler(FileSystemEventHandler):
    def __init__(self, signal):
        self.signal = signal

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.wav'):
            self.signal.emit(event.src_path)

class FolderWatcher(object):
    file_detected = Signal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.observer = Observer()

    def start_watching(self):
            event_handler = WavHandler(self.file_detected)
            self.observer.schedule(event_handler, self.folder_path, recursive=False)
            self.observer.start()

    def stop_watching(self):
        self.observer.stop()
        self.observer.join()


def compile_shader(src, stype):
    s = glCreateShader(stype)
    glShaderSource(s, src)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s

def create_test_data():
    """ Creates a 2D cross pattern with values from 0 to 32767 """
    points = []
    for x in np.linspace(-0.5, 0.5, 50):
        points.append([x, 0.0, (x + 0.5) * 32767])
    
    for y in np.linspace(-0.5, 0.5, 50):
        points.append([0.0, y, (y + 0.5) * 32767])
        
    return np.array(points, dtype=np.float32)
