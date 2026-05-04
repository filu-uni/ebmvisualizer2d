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
import polars.selectors as cs

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



#need to create them sorted after mesh and then x and y
def create_arrow_from_wav(file_path, number, out_folder="arrow_files", stride=1):
    out_dir = Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Layer_{number}.arrow"

    samplerate, data = wavfile.read(file_path)
    
    #values = np.mean([data[::stride,0],data[::stride,1],data[::stride,2],data[::stride,3]],axis=0).astype(np.float32)

    df = pl.DataFrame({
        "x": data[::stride, -4].astype(np.float32),
        "y": data[::stride, -3].astype(np.float32),
        "channel 1" :data[::stride,0],
        "channel 2" :data[::stride,1],
        "channel 3" :data[::stride,2],
        "channel 4" :data[::stride,3],
        })
    

    #df = df.group_by(["x","y"])
    #df = df.agg(pl.mean("channel 1","channel 2"
    df.write_ipc(out_file)

    del data
    
    del df
    
    print(f"Exported to {out_file}")
    return out_file

def get_df_from_arrow(file, ch="all", strategy="median", nth=4):
    file_path = Path(file).absolute()
    ldf = pl.scan_ipc(file_path).gather_every(nth)


    topo_formulas = {
        "Topo_A": pl.col("channel 4") - pl.col("channel 1"),
        "Topo_B": pl.col("channel 3") - pl.col("channel 2"),
        "Topo_C": pl.col("channel 4") + pl.col("channel 3") - (pl.col("channel 1") + pl.col("channel 2")),
        "Topo_D": pl.col("channel 4") + pl.col("channel 2") - (pl.col("channel 1") + pl.col("channel 1")),
    }


    if ch == "mean":
            # Calculate the horizontal mean across the channel columns for each row
            return ldf.select([
                pl.col("x"),
                pl.col("y"),
                # This averages the 4 values at the exact same x,y coordinate
                pl.mean_horizontal(r"^channel \d+$").alias("value").cast(pl.Float32)
            ])

    if ch == "all":
        # Turn channel 1, 2, 3, 4 columns into a single 'value' column
        return (
            ldf.unpivot(
                index=["x", "y"],
                on=[r"^channel \d+$"],
                value_name="value"
            )
            .select(["x", "y", "value"])
        )
    
    # Otherwise, handle specific Topo or named channels
    val_expr = topo_formulas.get(ch, pl.col(ch))
    return ldf.select([
        pl.col("x"),
        pl.col("y"),
        val_expr.alias("value")
    ])

def normalize_data(ldf, ch):
    # ch is expected to be ["value", "amount"]
    
    # 1. Define the normalization expression
    # Using (value - min) / (max - min) * 2 - 1 to get the -1 to 1 range
    def min_max_norm(col_name):
        c = pl.col(col_name)
        return (2 * (c - c.min()) / (c.max() - c.min()) - 1)

    # 2. Apply and select everything at once
    return ldf.select([
        min_max_norm("x").alias("x"), # Overwrite x with normalized x
        min_max_norm("y").alias("y"), # Overwrite y with normalized y
        pl.col(ch).cast(pl.Float32),               
    ])


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
                    (pl.col("value").cast(pl.Float32)).alias("bin"))
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
        if not self.files: return

        # 1. Combine all files into one plan
        ld = pl.concat([
            get_df_from_arrow(f, self.ch, self.strategy, self.nth) 
            for f in self.files
        ])

        # 2. Histogram (Optional: still groups by the 'value' itself)
        hist_arr = (
            ld.group_by("value")
            .agg(pl.len().alias("count"))
            .sort("value")
            .collect()
            .to_numpy()
        )
        self.carrier.histogram_finished.emit(hist_arr)
        del hist_arr

        # 3. Aggregate by Coordinate
        # This collapses multiple hits on the same (x,y) into one row
        strategies = {
            "max": pl.col("value").max(),
            "median": pl.col("value").median(),
            "mean": pl.col("value").mean(),
            "std": pl.col("value").std(),
            "amount": pl.len().alias("value")
        }
        
        selected_agg = strategies.get(self.strategy, strategies["mean"])
        # Returns an integer

        df = (
            ld.group_by(["x", "y"])  # This preserves x and y columns
            .agg(
                selected_agg.alias("value"),
                pl.len().alias("amount") # Count of how many values were at this x,y
            )
            .sort("value", descending=True)
        )
        df = normalize_data(df,"value")
        df = df.collect()
        self.carrier.finished.emit(df.to_numpy())
        del df

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
        if not event.is_directory and event.src_path.lower().endswith('.wav'):
            self.signals.file_ready.emit(event.src_path)

    def on_moved(self, event):
        # Handle 'Atomic Saves': Temp file is moved to final destination.
        if not event.is_directory and event.src_path.lower().endswith('.wav'):
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
