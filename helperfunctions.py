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


#need to create them sorted after mesh and then x and y
def create_arrow_from_wav(file_path, number, out_folder="arrow_files", stride=1):
    out_dir = Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Layer_{number}.arrow"

    samplerate, data = wavfile.read(file_path)
    
    values = np.mean([data[::stride,0],data[::stride,1],data[::stride,2],data[::stride,3]],axis=0).astype(np.float32)

    df = pl.DataFrame({
        "x": data[::stride, -4].astype(np.float32),
        "y": data[::stride, -3].astype(np.float32),
        "channel 1" :data[::stride,0],
        "channel 2" :data[::stride,1],
        "channel 3" :data[::stride,2],
        "channel 4" :data[::stride,3],
        "mean" : values
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


class HistogramSignals(QObject):
    filteredHistogram = Signal(object)

class HistogramFilterTask(QRunnable):
    
    def __init__(self, ch, files):
        super().__init__()
        self.ch = ch
        self.files = files
        self.signals = HistogramSignals()

    def run(self):
            
            # 1. Create a list of all LazyFrames
            # This just stores the "instructions" for each file, using almost no RAM
            lazy_plans = [
                get_df_from_arrow(file, self.ch,1) 
                for file in self.files
            ]
            

            hist_list = [ df.select(
                    (pl.col(self.ch).cast(pl.Float32)).alias("bin"))
                    for df in lazy_plans]

            hist_list = [ ( 
                df.group_by("bin")
                .agg(pl.len().alias("count")) # pl.len() is the most efficient way to count rows
                .sort("bin")
                )
                for df in hist_list ]

            dfs_with_ids = [
                df.with_columns(pl.lit(f"hist_{i}").alias("hist_id"))
                for i, df in enumerate(hist_list)
            ]

            combined_df = pl.concat(dfs_with_ids)

            # Define how "aggressive" you want to be in finding differences
            # A higher threshold means only very big differences are caught
            STDEV_THRESHOLD = 0.4 

            interesting_spots = (
                combined_df.group_by("bin")
                .agg(
                    pl.col("count").std().alias("std_diff"),
                    pl.col("count").max() - pl.col("count").min().alias("range_diff"),
                    pl.col("count").mean().alias("avg_val")
                )
                .filter(pl.col("std_diff") > (pl.col("avg_val") * STDEV_THRESHOLD)) # Example: 40% deviation from mean
                .sort("std_diff", descending=True)
            )

            final_histogram = combined_df.join(
                interesting_spots.select("bin"), 
                on="bin"
            ).sort("count",descending=True)

            hist = final_histogram.collect()
            histogram = hist.to_numpy()

            self.signals.filteredHistogram.emit(histogram)
            print(histogram)
    


class DataCarriage(QObject):
    finished = Signal(object)
    histogram_finished = Signal(object)

class DataWorker(QRunnable):

    def __init__(self, nth, ch, files, strategy="mean"):
        super().__init__()
        self.nth = nth
        self.ch = ch
        self.files = files
        self.carrier = DataCarriage()
        self.strategy = strategy

    def run(self):
            
            if len(self.files) < 1:
                return
            
            # 1. Create a list of all LazyFrames
            # This just stores the "instructions" for each file, using almost no RAM
            lazy_plans = [
                get_df_from_arrow(file, self.ch, self.nth) 
                for file in self.files
            ]
            
            #we devide by n since we will add n measurements back on top again
            n = len(self.files)

            # 2. Concat them all at once
            # Polars can now optimize the entire operation globally
            
            ldf = pl.concat(lazy_plans,rechunk=True) if n > 1 else lazy_plans[0]

            histogram = (
                ldf.group_by(self.ch)
                .agg(pl.len().alias("amount")) # pl.len() is the most efficient way to count rows
                .sort(self.ch)
            )
            

            #noise_reduced = histogram.select(
            #        (pl.col("amount") > 1000).alias("reduced"),
            #        pl.col(self.ch)
            #        )

            ldf = ldf.join(histogram, on=self.ch,how="semi")
    
            histdf = histogram.collect()
            # Convert to 2D numpy array: [[energy1, count1], [energy2, count2], ...]
            hist = histdf.to_numpy()
            self.carrier.histogram_finished.emit(hist)

            temp = ldf.select(pl.col(self.ch)).collect()

            ldf = ldf.select(
                    pl.col("x"),
                    pl.col("y"),
                    (pl.col(self.ch)).alias("value"))
            
            if self.strategy == "max":
                ldf = ldf.group_by(["x", "y"]).agg([pl.col("value").max()]).sort("value",descending=True)
            else:
                ldf = ldf.group_by(["x", "y"]).agg([pl.col("value").mean()]).sort("value",descending=True)

            ldf = normalize_data(ldf,"value")

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
