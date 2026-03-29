from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
#from collections import Counter
import os
from pathlib import Path
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PySide6.QtCore import Signal, QThread, QRunnable, QThreadPool, QObject
from PySide6.QtWidgets import QApplication
import pyarrow
import polars as pl
from natsort import natsorted
from watchdog.events import FileSystemEventHandler, FileClosedEvent
import time


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


def create_histogram_from_arrow_folder(folder_path, ch="mean"):
    folder = Path(folder_path)
    pattern = str(folder / "*.arrow")
    
    hist_df = (
    pl.scan_ipc(pattern)
    .select(ch)
    .group_by(ch)
    .len()
    .sort(ch) 
    .collect()
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
    
    stride = 8
    mean = np.mean([data[::stride,0],data[::stride,1],data[::stride,2],data[::stride,3]],axis=0)
    df = pl.DataFrame({
        "x": data[:, -4].astype(np.float32),
        "y": data[:, -3].astype(np.float32),
        "mean": mean
        })

    df.write_ipc(out_file)

    del data
    
    del df
    
    print(f"Exported to {out_file}")
    return out_file


def get_df_from_arrow(file, ch="mean", nth=8):
    
    file_path = Path(file).absolute()
    
    ldf = pl.scan_ipc(file_path)
    
    if nth > 1:
        ldf = ldf.gather_every(nth)
    
    ldf = ldf.select(["x", "y", ch])
    
    return ldf

def normalize_data(ldf, ch):
    # Compute min/max lazily
    x_min = pl.col("x").min()
    x_max = pl.col("x").max()
    y_min = pl.col("y").min()
    y_max = pl.col("y").max()

    # Add normalized columns lazily
    ldf = ldf.with_columns([
        ((2 * (pl.col("x") - x_min) / (x_max - x_min)) - 1).alias("x_norm"),
        ((2 * (pl.col("y") - y_min) / (y_max - y_min)) - 1).alias("y_norm"),
    ])

    # Select only relevant columns
    ldf = ldf.select(["x_norm", "y_norm", ch])
    return ldf



class DataCarriage(QObject):
    finished = Signal(object)

class DataWorker(QRunnable):

    def __init__(self, nth, ch, files):
        super().__init__()
        self.nth = nth
        self.ch = ch
        self.files = files
        self.carrier = DataCarriage(QApplication.instance())

    def run(self):
            
            # 1. Create a list of all LazyFrames
            # This just stores the "instructions" for each file, using almost no RAM
            lazy_plans = [
                get_df_from_arrow(file, self.ch, self.nth) 
                for file in self.files
            ]
            
            # 2. Concat them all at once
            # Polars can now optimize the entire operation globally
            ldf = pl.concat(lazy_plans,rechunk=True)

            ldf = normalize_data(ldf, self.ch)

            df = ldf.collect()

            arr = df.to_numpy()

            self.carrier.finished.emit(arr)


class ArrowFileCreatorSignals(QObject):
    finishedTask = Signal()

class CreateArrowFile(QRunnable):
    def __init__(self,file,number,out_path):
        super().__init__()
        self.file = file
        self.number = number
        self.out_path = out_path
        self.signal = ArrowFileCreatorSignals()

    def run(self):
            
        create_arrow_from_wav(self.file,self.number,self.out_path)
        print(f"Layer {self.number} created")
        self.signal.finishedTask.emit()
            

class WatchdogSignals(QObject):
    startWatching = Signal(str)
    stopWatching = Signal()
    file_ready = Signal(str)
    error = Signal(str)

class WatchdogObserver(FileSystemEventHandler):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals

    def on_closed(self, event):
        # IN_CLOSE_WRITE: The file descriptor is released after writing.
        if not event.is_directory:
            self.signals.file_ready.emit(event.src_path)

    def on_moved(self, event):
        # Handle 'Atomic Saves': Temp file is moved to final destination.
        if not event.is_directory:
            self.signals.file_ready.emit(event.src_path)

class AsyncWatchdogTask(QRunnable):
    """
    The background task managed by QThreadPool.
    """
    def __init__(self, watch_path):
        super().__init__()
        self.watch_path = watch_path
        self.signals = WatchdogSignals()
        self.observer = Observer()
        self._keep_running = True

    def run(self):
        try:
            handler = WatchdogObserver(self.signals)
            self.observer.schedule(handler, self.watch_path, recursive=False)
            self.observer.start()

            # Signal that monitoring has officially begun
            self.signals.startWatching.emit(self.watch_path)

            # Keep the QRunnable alive while monitoring
            while self._keep_running:
                time.sleep(0.1)

            self.observer.stop()
            self.observer.join()
            self.signals.stopWatching.emit()

        except Exception as e:
            self.signals.error.emit(str(e))

    def stop(self):
        """Called from the main thread to shut down the watcher."""
        self._keep_running = False


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
