
# Copyright (C) 2013 Riverbank Computing Limited.
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
from __future__ import annotations

import sys
from PySide6.QtCore import Qt, Signal, QThread, QDir, QPoint, QPointF, QMargins
from PySide6.QtGui import QSurfaceFormat, QMovie, QPainter

from PySide6.QtWidgets import QApplication,QSlider, QHBoxLayout,QVBoxLayout, QWidget, QLabel, QPushButton, QSpinBox, QComboBox, QFileDialog, QStackedLayout
from PySide6.QtCharts import QChart, QChartView, QBarSet, QLineSeries, QBarCategoryAxis, QValueAxis

from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from superqt import QRangeSlider
from pathlib import Path
import helperfunctions as helpers
import numpy as np
import openglwidget as glw

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
        #self.value_label.editingFinished.connect(self.finishedEditing)

        self.slider.valueChanged.connect(self.update_label)
        self.slider.valueChanged.connect(self.sendValue)
        self.slider.sliderReleased.connect(self.finishedEditing)


    def getValue(self):
        return self.slider.value()
    def update_label(self, value):
        self.value_label.blockSignals(True)
        value = value
        self.value_label.setValue(value)
        self.value_label.blockSignals(False)
    def update_slider(self, value):
        value = value
        self.slider.setValue(value)
        self.valueChanged.emit()
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

        self.series = QLineSeries()

        self.chart = QChart()
        self.chart.addSeries(self.series)
        self.chart.legend().hide() 
        #self.chart.layout().setContentsMargins(0, 0, 0, 0)
        #self.chart.setMargins(QMargins(10, 0, 30, 25))#this is ugly code so it looks nice. will have to be changed
        #self.chart.setBackgroundRoundness(0)


        self.axis_x = QValueAxis()
        self.axis_y = QValueAxis()

        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)

        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)

        self.chart_view = QChartView(self.chart,parent=self)

        self.tooltip = QLabel(self.chart_view)
        self.tooltip.setStyleSheet("background: rgba(255, 255, 255, 180); border: 1px solid black; padding: 2px;")
        self.tooltip.hide()


        self.axis_y.setLabelsVisible(False)
        self.axis_x.setLabelsVisible(False)
        self.axis_x.setGridLineVisible(False)
        self.axis_y.setGridLineVisible(False)
        self.axis_x.setLineVisible(False)
        self.axis_y.setLineVisible(False)

        self.chart_view.setMouseTracking(True)
        self.chart_view.mouseMoveEvent = self.on_mouse_move
        self.chart_view.setRenderHint(QPainter.Antialiasing) 
        self.chart_view.show()


        self.layout.addWidget(self.chart_view)


    def update_data(self, hist_df,ch):
        x_values = hist_df[ch].to_list()
        y_values = hist_df["len"].to_list()
        points = [QPointF(float(x), float(y)) for x, y in zip(x_values, y_values)]
        
        self.series.replace(points)
        
        if x_values and y_values:
            self.axis_x.setRange(min(x_values), max(x_values))
            self.axis_y.setRange(0, max(y_values) * 1.1)

    def on_mouse_move(self, event):
        if self.chart.plotArea().contains(event.position()):
            data_point = self.chart.mapToValue(event.position(), self.series)

            x_val = data_point.x()

            self.tooltip.setText(f"X: {x_val:.0f}")
            self.tooltip.move(event.position().toPoint() + QPoint(15, -15))
            self.tooltip.show()
        else:
            self.tooltip.hide()

        super(QChartView, self.chart_view).mouseMoveEvent(event)


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
        self.layer = 1
        self.energy_range = (1000,4000)
        self.resolution = 10
        self.pointsize = 3
        self.channel = "channel_1"
        self.widgets = dict()
        self.folder = QDir()
        self.arrow_folder = QDir("arrow_files")



        self.folder_button = QPushButton(self)
        self.folder_button.setText("Choose Wav File Folder")
        
        self.arrow_button = LoadingButton(parent=self,text="create arrow Files")
        #self.arrow_button = QPushButton(self)
        #self.arrow_button.setText("create arrow Files")

        self.recalculate = LoadingButton(parent=self,text="Recalculate")

        self.resolutionwidget = SliderWidget("Resolution",(1,1000),1)

        self.channelwidget = QComboBox()
        self.channelwidget.addItems(["channel_1","channel_2","channel_3","channel_4"])

        self.pointsizewidget = SliderWidget("PointSize",(1,20),3)

        self.histogramWidget = HistogramPlot(self)
        self.histogramWidget.setMinimumSize(200,200)
        #self.histogramWidget.setMaximumSize(200,200)
        self.histogramWidget.show()

        self.layerwidget = SliderWidget("Layers",(1,100),1)
        self.energywidget = SliderWidget("Energy",(1,2**15),(1000,4000),double=True)

        self.layer_display = QLabel()
        self.layer_display.setText("")
        
        self.export_button = QPushButton()
        self.export_button.setText("Export to Png")



        self.folder_button.released.connect(self.choose_folder)
        self.arrow_button.released.connect(self.create_arrow_files)
        self.recalculate.released.connect(self.beginRecalculation)
        self.channelwidget.currentTextChanged.connect(self.beginRecalculation)
        self.pointsizewidget.valueChanged.connect(self.get_pointsize)
        self.energywidget.valueChanged.connect(self.get_energy_range)
        self.layerwidget.released.connect(self.beginRecalculation)
        self.resolutionwidget.released.connect(self.beginRecalculation)
        self.export_button.released.connect(self.export.emit)

        layout.addWidget(self.folder_button)
        layout.addWidget(self.arrow_button)

        energyLayout = QVBoxLayout()
        energyLayout.addWidget(self.histogramWidget)
        energyLayout.addWidget(self.energywidget)
        layout.addLayout(energyLayout)

        layout.addWidget(self.pointsizewidget)
        layout.addWidget(self.resolutionwidget)
        layout.addWidget(self.channelwidget)
        layout.addWidget(self.layerwidget)
        layout.addWidget(self.layer_display)
        layout.addWidget(self.recalculate)
        layout.addWidget(self.export_button)


        if helpers.get_arrow_files(self.arrow_folder.absolutePath()):
            self.updateHistogram()
        layout.addStretch()


    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")

        if folder:  
            self.folder = QDir(folder)
            self.folder_button.setText(self.folder.absolutePath())
        else:
            self.folder_button.setText("Choose Wav File Folder")

    def create_arrow_files(self):
        #self.arrow_button.start_loading()
        wav_files = helpers.get_wav_files(self.folder.absolutePath())
        counter = 0
        self.layerwidget.setRange((1,len(wav_files)))
        for file in wav_files:
            counter += 1
            helpers.create_arrow_from_wav(file,counter,self.arrow_folder.absolutePath())
            print(f"Layer {counter} created")
        #self.updateHistogram()
        #self.arrow_button.stop_loading()


    def get_energy_range(self):
        self.energy_range = self.energywidget.getValue()
        self.energyChanged.emit(self.energy_range)
    def get_pointsize(self):
        self.pointsize = self.pointsizewidget.getValue()
        self.pointsizeChanged.emit(self.pointsize)
    def getChannel(self):
        return self.channel
    def getResolution(self):
        return self.resolution
    def getLayer(self):
        return self.layer
    def getCsvFolder(self):
        return self.arrow_folder
    def updateHistogram(self):
        hist_df = helpers.create_histogram_from_arrow_folder(self.arrow_folder.absolutePath(),self.channel)
        self.histogramWidget.update_data(hist_df,self.channel)
        self.energywidget.setRange((hist_df[self.channel].min(),hist_df[self.channel].max()))
    def updateLayers(self):
        layers = len(helpers.get_arrow_files(self.arrow_folder.absolutePath()))
        self.layerwidget.setRange((1,layers))
    def beginRecalculation(self):
        self.updateLayers() 
        self.layer = self.layerwidget.getValue()
        if self.folder.exists():
            wav_folder = helpers.get_wav_files(self.folder.absolutePath())
            if self.layer < len(wav_folder):
                self.layer_display.setText(str(wav_folder[self.layer].name))
        self.resolution = self.resolutionwidget.getValue()
        self.channel = self.channelwidget.currentText()
        self.begincalculation.emit()
    def startCalculation(self):
        self.recalculate.start_loading()
    def finishCalculation(self):
        self.recalculate.stop_loading()

