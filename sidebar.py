
# Copyright (C) 2013 Riverbank Computing Limited.
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
from __future__ import annotations

import sys
from PySide6.QtCore import Qt, Signal, QThread, QDir, QPoint, QPointF, QMargins, QRunnable, QThreadPool

from PySide6.QtGui import QSurfaceFormat, QMovie, QPainter, QColor, QGradient, QLinearGradient

from PySide6.QtWidgets import QApplication,QSlider, QHBoxLayout,QVBoxLayout, QWidget, QLabel, QPushButton, QSpinBox, QComboBox, QFileDialog, QStackedLayout
from PySide6.QtCharts import QChart, QChartView, QBarSet, QAreaSeries, QLineSeries, QBarCategoryAxis, QValueAxis, QScatterSeries

from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from superqt import QRangeSlider
from pathlib import Path
import helperfunctions as helpers
import numpy as np
import openglwidget as glw
import polars as pl
import os

class LoadingButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        
        self.original_text = text
        self.movie = QMovie("loading_bar.gif")

        self.overlay = QLabel(self) 
        self.overlay.resize(self.size())
        self.overlay.setMovie(self.movie)
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay.hide()

    def resizeEvent(self, event):
        self.overlay.resize(self.size())
        super().resizeEvent(event)

    def start_loading(self):
        self.setText("") 
        self.overlay.show()
        self.movie.start()
        self.setEnabled(False)

    def start_non_blocking_loading(self):
        self.setText("") 
        self.overlay.show()
        self.movie.start()
    
    def isRunning(self):
        return self.movie.state() == QMovie.MovieState.Running

    def stop_loading(self):
        self.movie.stop()
        self.overlay.hide()
        self.setText(self.original_text)
        self.setEnabled(True)




class RangeSpinBox(QWidget):
    valueChanged = Signal(object)
    editingFinished = Signal()
    def __init__(self,value_range,init_val):
        super().__init__()
        self.value = init_val
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        self.spinboxmin = QSpinBox()
        self.spinboxmin.setRange(value_range[0],value_range[1])
        self.spinboxmin.setValue(init_val[0])
        self.spinboxmax = QSpinBox()
        self.spinboxmax.setRange(value_range[0],value_range[1])
        self.spinboxmax.setValue(init_val[1])
        
        self.layout.addWidget(self.spinboxmin)
        self.layout.addWidget(self.spinboxmax)
        
        self.spinboxmin.valueChanged.connect(self.update_value_min)
        self.spinboxmax.valueChanged.connect(self.update_value_max)

        self.spinboxmin.editingFinished.connect(self.finishedEditing)
        self.spinboxmax.editingFinished.connect(self.finishedEditing)

    def update_value_min(self,value):
        self.value = (value,self.value[1])
        self.valueChanged.emit(self.value)
    def update_value_max(self,value):
        self.value = (self.value[0],value)
        self.valueChanged.emit(self.value)
    def setRange(self,value_range_min,value_range_max):
        self.spinboxmin.setRange(value_range_min,value_range_max)
        self.spinboxmax.setRange(value_range_min,value_range_max)
    def setValue(self,value):
        self.spinboxmin.setValue(value[0])
        self.spinboxmax.setValue(value[1])
    def finishedEditing(self):
        self.editingFinished.emit()

class SliderWidget(QWidget):
    valueChanged = Signal()
    released = Signal()
    """A single slider with a name above and value spin box below. If double = True it will be a range slider instead"""
    def __init__(self, name, val_range, init_val,double = False):
        super().__init__()
        
        self.double = double
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.name_label = QLabel(name)


        if double:
            self.slider = QRangeSlider(Qt.Horizontal)
        else:
            self.slider =  QSlider(Qt.Horizontal)
       
        self.slider.setRange(val_range[0],val_range[1])
        self.slider.setValue(init_val)


        if self.double:
            self.value_label = RangeSpinBox(val_range,init_val)
        else:
            self.value_label = QSpinBox()
            self.value_label.setValue(init_val)
            self.value_label.setRange(val_range[0],val_range[1])

        layout.addWidget(self.name_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label, alignment=Qt.AlignCenter)

        self.value_label.valueChanged.connect(self.update_slider)
        self.value_label.valueChanged.connect(self.finishedEditing)
        self.value_label.editingFinished.connect(self.finishedEditing)

        self.slider.valueChanged.connect(self.update_label)
        self.slider.valueChanged.connect(self.sendValue)
        self.slider.sliderReleased.connect(self.finishedEditing)


    def getValue(self):
        return self.slider.value()
    def setValue(self,value):
        self.update_label(value)
        self.update_slider(value)
        self.released.emit()
    def update_label(self, value):
        self.value_label.blockSignals(True)
        value = value
        self.value_label.setValue(value)
        self.value_label.blockSignals(False)
    def update_slider(self, value):
        self.slider.blockSignals(True)
        value = value
        self.slider.setValue(value)
        self.valueChanged.emit()
        self.slider.blockSignals(False)
    def sendValue(self):
        self.valueChanged.emit()
    def finishedEditing(self):
        self.released.emit()
    def setRange(self,new_range):
        self.slider.setRange(new_range[0],new_range[1])
        self.value_label.setRange(new_range[0],new_range[1])


class HistogramPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 1. Use QLineSeries for the "outline"
        self.line_series = QLineSeries()
        
        # 2. Wrap it in a QAreaSeries to fill the bottom
        self.area_series = QAreaSeries(self.line_series)
        
        # Styling the fill
        fill_color = QColor(52, 152, 219, 150) # Semi-transparent blue
        self.area_series.setColor(fill_color)
        self.area_series.setBorderColor(QColor(52, 152, 219)) # Solid blue border

        self.chart = QChart()
        self.chart.addSeries(self.area_series)
        self.chart.legend().hide()
        self.chart.layout().setContentsMargins(0, 0, 0, 0)
        self.chart.setBackgroundVisible(False)

        # 3. Use ValueAxis for both (faster than CategoryAxis)
        self.axis_x = QValueAxis()
        self.axis_y = QValueAxis()
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        self.area_series.attachAxis(self.axis_x)
        self.area_series.attachAxis(self.axis_y)

        # Hide visual clutter
        for ax in [self.axis_x, self.axis_y]:
            ax.setVisible(False)

        self.view = QChartView(self.chart)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.layout.addWidget(self.view)

    def update_data(self, data_array):
        """Expects 2D numpy array [[energy, count], ...]"""
        if data_array is None or len(data_array) == 0:
            return

        # Prepare points (High speed: QLineSeries.replace is faster than clearing)
        points = [QPointF(row[0], row[1]) for row in data_array]
        self.line_series.replace(points)

        # Update Ranges
        self.axis_x.setRange(data_array[:, 0].min(), data_array[:, 0].max())
        self.axis_y.setRange(0, data_array[:, 1].max() * 1.05)

class Sidebar(QWidget):
    begincalculation = Signal()
    energyChanged = Signal(object)
    pointsizeChanged = Signal(object)
    export = Signal()
    """Vertical sidebar with multiple sliders"""
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layer = (1,1)
        self.energy_range = (1000,4000)
        self.resolution = 8
        self.pointsize = 3
        self.channel = "mean"
        self.widgets = dict()
        self.wav_folder = QDir()
        self.arrow_folder = QDir("arrow_files")
        self.arrowpool = QThreadPool(self)
        self.watchdog_file_counter = 0



        self.wav_folder_button = QPushButton(self)
        self.wav_folder_button.setText("Choose Wav File Folder")
        self.arrow_folder_button = QPushButton(self)
        self.arrow_folder_button.setText("Choose Arrow File Folder")
        
        self.arrow_button = LoadingButton(parent=self,text="create arrow Files")

        self.watchdog = LoadingButton(parent=self,text="deploy watchdog")

        self.recalculate = LoadingButton(parent=self,text="Recalculate")


        self.channelwidget = QComboBox()
        self.channelwidget.addItems(["mean"])

        self.resolutionwidget = QSpinBox()
        self.resolutionwidget.setMinimum(1)
        self.resolutionwidget.setValue(4)
        
        #self.resolutionwidget = SliderWidget("Resolution",(1,40),8)

        self.pointsizewidget = QSpinBox()
        self.pointsizewidget.setMinimum(1)
        self.pointsizewidget.setValue(3)
        #self.pointsizewidget = SliderWidget("PointSize",(1,20),3)

        self.histogramWidget = HistogramPlot(self)
        self.histogramWidget.setMinimumSize(200,300)
        self.histogramWidget.show()

        self.layerwidget = SliderWidget("Layers",(1,100),(1,1),double=True)
        self.energywidget = SliderWidget("Energy",(1,2**15),(1000,4000),double=True)

        self.layer_display = QLabel()
        self.layer_display.setText("")
        
        self.export_button = QPushButton()
        self.export_button.setText("Export to Png")



        self.wav_folder_button.released.connect(self.choose_wav_folder)
        self.arrow_folder_button.released.connect(self.choose_arrow_folder)
        self.arrow_button.released.connect(self.create_arrow_files)
        self.watchdog.released.connect(self.flip_watchdog)

        self.recalculate.released.connect(self.beginRecalculation)
        self.channelwidget.activated.connect(self.beginRecalculation)
        self.pointsizewidget.valueChanged.connect(self.get_pointsize)
        self.energywidget.valueChanged.connect(self.get_energy_range)
        self.layerwidget.released.connect(self.beginRecalculation)
        self.resolutionwidget.valueChanged.connect(self.beginRecalculation)
        self.export_button.released.connect(self.export.emit)

        layout.addWidget(self.wav_folder_button)
        layout.addWidget(self.arrow_folder_button)
        layout.addWidget(self.watchdog)
        layout.addWidget(self.arrow_button)

        energyLayout = QVBoxLayout()
        energyLayout.addWidget(self.histogramWidget)
        energyLayout.addWidget(self.energywidget)
        layout.addLayout(energyLayout)

        pointsLayout = QHBoxLayout()
        pointsLayout.addWidget(self.pointsizewidget)
        pointsLayout.addWidget(self.resolutionwidget)
        layout.addLayout(pointsLayout)

        layout.addWidget(self.channelwidget)
        layout.addWidget(self.layerwidget)
        layout.addWidget(self.layer_display)
        layout.addWidget(self.recalculate)
        layout.addWidget(self.export_button)


        layout.addStretch()
        self.updateLayers()


    def choose_wav_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")

        if folder:  
            self.wav_folder = QDir(folder)
            self.wav_folder_button.setText(self.wav_folder.absolutePath())
        else:
            self.wav_folder_button.setText("Choose Wav File Folder")

    def choose_arrow_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")

        if folder:  
            self.arrow_folder = QDir(folder)
            self.arrow_folder_button.setText(self.arrow_folder.absolutePath())

            files = self.arrow_folder.entryInfoList(filters=QDir.Filter.Files,sort=QDir.SortFlag.Name)
            if files:
                self.channelwidget.clear()
                file = files[0].absolutePath()
                column_names = pl.scan_ipc(file).collect_schema().names()
                self.channelwidget.addItems(column_names[2:])
            self.updateLayers()
        else:
            self.arrow_folder_button.setText("Choose Arrow File Folder")
    
    def create_arrow_file(self,file):
        if os.path.isfile(file) and file.endswith(".wav"):
            self.watchdog_file_counter += 1
            task = helpers.CreateArrowFile(file,self.watchdog_file_counter,self.arrow_folder.absolutePath())
            task.signal.finishedTask.connect(self.updateLayers)
            self.arrowpool.start(task)
            

    def create_arrow_files(self):
        wav_files = helpers.get_wav_files(self.wav_folder.absolutePath())
        counter = 0
        self.layerwidget.setRange((1,len(wav_files)))

        for file in wav_files:
            counter += 1
            task = helpers.CreateArrowFile(file,counter,self.arrow_folder.absolutePath())
            self.arrowpool.start(task)


    def flip_watchdog(self):
        if self.watchdog.isRunning():
            self.watchdog_task.stop()
            self.watchdog.stop_loading()
        else:
            self.watchdog.start_non_blocking_loading()
            self.watchpool = QThreadPool(self)
            self.watchdog_task = helpers.AsyncWatchdogTask(self.wav_folder.absolutePath())
            self.watchdog_task.signals.file_ready.connect(self.create_arrow_file)
            self.watchdog_task.signals.error.connect(lambda e: print(f"Error: {e}"))

            self.watchdog.setText("watching" + self.arrow_folder.absolutePath())
            self.watchpool.start(self.watchdog_task)

    def get_energy_range(self):
        self.energy_range = self.energywidget.getValue()
        self.energyChanged.emit(self.energy_range)
    def get_pointsize(self):
        self.pointsize = self.pointsizewidget.value()
        self.pointsizeChanged.emit(self.pointsize)
    def getChannel(self):
        return self.channel
    def getResolution(self):
        return self.resolution
    def getLayer(self):
        return self.layer
    def getArrowFolder(self):
        return self.arrow_folder
    def updateHistogram(self,hist):
        self.histogramWidget.update_data(hist)
        self.energywidget.setRange((hist[0].min(),hist[0].max()))
   
    def updateLayers(self):
        layers = len(helpers.get_arrow_files(self.arrow_folder.absolutePath()))
        self.layerwidget.setRange((1,layers))
        lowerbound = layers - 10 if layers - 10 >= 1 else 1 
        self.layerwidget.setValue((lowerbound,layers))
        

    def beginRecalculation(self):
        self.layer = self.layerwidget.getValue()
        if self.wav_folder.exists():
            wav_folder = helpers.get_wav_files(self.wav_folder.absolutePath())
            if self.layer[1] < len(wav_folder):
                self.layer_display.setText(str(wav_folder[self.layer[1]].name))
        self.resolution = self.resolutionwidget.value()
        self.channel = self.channelwidget.currentText()
        self.begincalculation.emit()
    def startCalculation(self):
        self.recalculate.start_loading()
    def finishCalculation(self):
        self.recalculate.stop_loading()

