
# Copyright (C) 2013 Riverbank Computing Limited.
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
from __future__ import annotations

import sys
from PySide6.QtCore import Qt, Signal, QThread, QDir, QPoint, QPointF, QMargins, QRunnable, QThreadPool

from PySide6.QtGui import QSurfaceFormat, QMovie, QPainter, QColor, QGradient, QLinearGradient, QPen

from PySide6.QtWidgets import QApplication,QSlider, QHBoxLayout, QVBoxLayout, QGridLayout, QWidget, QLabel, QPushButton, QSpinBox, QComboBox, QFileDialog, QStackedLayout
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
from natsort import natsorted
import pint
from CustomWidgets import LoadingButton, RangeSpinBox, SliderWidget, HistogramPlot 

        
class Sidebar(QWidget):
    begincalculation = Signal()
    energyChanged = Signal(object)
    amountChanged = Signal(object)
    pointsizeChanged = Signal(object)
    export = Signal()
    """Vertical sidebar with multiple sliders"""
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.layer = (1,1)
        self.energy_range = (1000,4000)
        self.amount_range = (-1000,10000)
        self.resolution = 8
        self.pointsize = 3
        self.channel = "mean"
        self.strategy = "mean"
        self.widgets = dict()
        self.wav_folder = QDir()
        self.arrow_folder = QDir("arrow_files")
        self.arrowpool = QThreadPool(self)
        self.histopool = QThreadPool(self)
        self.watchdog_file_counter = 0
        self.Units = pint.UnitRegistry()



        self.wav_folder_button = QPushButton(self)
        self.wav_folder_button.setText("Choose Wav File Folder")
        self.arrow_folder_button = QPushButton(self)
        self.arrow_folder_button.setText("Choose Arrow File Folder")
        
        self.arrow_button = LoadingButton(parent=self,text="create arrow Files")

        self.watchdog = LoadingButton(parent=self,text="deploy watchdog")
        self.histoFilter = LoadingButton(parent=self,text="calculate interesting frequencies")

        self.recalculate = LoadingButton(parent=self,text="Recalculate")


        self.channelwidget = QComboBox()
        

        self.aggregationWidget = QComboBox()
        self.aggregationWidget.addItems(["mean","max","median","std"])

        self.resolutionwidget = QSpinBox()
        self.resolutionwidget.setMinimum(1)
        self.resolutionwidget.setValue(4)

        self.pointsizewidget = QSpinBox()
        self.pointsizewidget.setMinimum(1)
        self.pointsizewidget.setValue(3)

        self.histogramWidget = HistogramPlot(self)
        self.histogramWidget.setMinimumSize(200,250)
        self.histogramWidget.show()

        self.layerwidget = SliderWidget("Layers",(1,100),(1,1),double=True)
        self.energywidget = SliderWidget("Energy",(-1000,2**15),(1000,4000),double=True)
        self.amountwidget = SliderWidget("Amount",(-1000,100000),(0,1000),double=True,orientation=Qt.Vertical)

        self.layer_display = QLabel()
        self.layer_display.setText("")
        self.position_display = QLabel()
        self.position_display.setText("")
        
        
        self.export_button = QPushButton()
        self.export_button.setText("Export to Png")


        self.wav_folder_button.released.connect(self.choose_wav_folder)
        self.arrow_folder_button.released.connect(self.choose_arrow_folder)
        self.arrow_button.released.connect(self.create_arrow_files)
        self.watchdog.released.connect(self.flip_watchdog)

        self.recalculate.released.connect(self.beginRecalculation)
        self.histoFilter.released.connect(self.filterHistogram)
        self.channelwidget.activated.connect(self.beginRecalculation)
        self.aggregationWidget.activated.connect(self.beginRecalculation)
        self.pointsizewidget.valueChanged.connect(self.get_pointsize)
        self.energywidget.valueChanged.connect(self.get_energy_range)
        self.energywidget.rangeAdjusted.connect(self.histogramWidget.updateRedBorderLinesEnergy)
        self.amountwidget.rangeAdjusted.connect(self.histogramWidget.updateRedBorderLinesAmount)
        self.amountwidget.valueChanged.connect(self.get_amount_range)
        self.amountwidget.valueChanged.connect(self.get_amount_range)
        self.histogramWidget.x_rangeChanged.connect(self.energywidget.setRange)
        self.histogramWidget.y_rangeChanged.connect(self.amountwidget.setRange)
        self.layerwidget.released.connect(self.beginRecalculation)
        self.resolutionwidget.valueChanged.connect(self.beginRecalculation)
        self.export_button.released.connect(self.export.emit)

        layout.addWidget(self.wav_folder_button)
        layout.addWidget(self.arrow_folder_button)
        layout.addWidget(self.watchdog)
        layout.addWidget(self.arrow_button)
        layout.addWidget(self.histoFilter)

        energyLayout = QGridLayout()
        energyLayout.addWidget(self.histogramWidget,0,0)
        energyLayout.addWidget(self.position_display,0,0,Qt.AlignTop | Qt.AlignLeft)
        energyLayout.addWidget(self.energywidget,1,0,1,2)
        energyLayout.addWidget(self.amountwidget,0,1)
        layout.addLayout(energyLayout)

        optionsLayout = QGridLayout()
        layout.addLayout(optionsLayout)
        optionsLayout.addWidget(self.pointsizewidget,0,0)
        optionsLayout.addWidget(QLabel("change the visual Point Size"),0,1)
        optionsLayout.addWidget(self.resolutionwidget,1,0)
        optionsLayout.addWidget(QLabel("change how many points are skipped"),1,1)
        optionsLayout.addWidget(self.channelwidget,2,0)
        optionsLayout.addWidget(QLabel("which channel should be shown"),2,1)
        optionsLayout.addWidget(self.aggregationWidget,3,0)
        optionsLayout.addWidget(QLabel("which aggregation strategy to use"),3,1)

        
        layout.addWidget(self.layerwidget)
        layout.addWidget(self.layer_display)
        lowest_layout = QHBoxLayout()
        layout.addLayout(lowest_layout)
        lowest_layout.addWidget(self.recalculate)
        lowest_layout.addWidget(self.export_button)

        self.histoFilter.setVisible(False)


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
                self.channelwidget.addItems(["all","mean"])
                file = files[0].absolutePath()
                column_names = pl.scan_ipc(file).collect_schema().names()
                self.channelwidget.addItems(column_names[2:])
                if "channel 1" in column_names[2:]:
                    self.channelwidget.addItems(["Topo_A","Topo_B","Topo_C","Topo_D"])
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
            self.watchdog_file_counter = 0
        else:
            amount = len(helpers.get_wav_files(self.wav_folder.absolutePath()))
            self.watchdog_file_counter = amount
            self.watchdog.start_non_blocking_loading()
            self.watchpool = QThreadPool(self)
            self.watchdog_task = helpers.AsyncWatchdogTask(self.wav_folder.absolutePath())
            self.watchdog_task.signals.file_ready.connect(self.create_arrow_file)
            self.watchdog_task.signals.error.connect(lambda e: print(f"Error: {e}"))

            self.watchpool.start(self.watchdog_task)

    def setPositionDisplay(self,pos):
        # 1. Convert your coordinates to micrometer
        x_mu = (pos[0] * self.Units.decimeter).to("millimeter")
        y_mu = (pos[1] * self.Units.decimeter).to("millimeter")

        # 2. Format with 3 decimals and short unit name (~)
        # .3f ensures 3 decimal places, ~ ensures 'µm' instead of 'micrometer'
        x_str = f"{x_mu:~.3f}"
        y_str = f"{y_mu:~.3f}"

        # 3. Update display
        self.position_display.setText(f"({x_str}, {y_str})")

    def get_energy_range(self):
        self.energy_range = self.energywidget.getValue()
        self.energyChanged.emit(self.energy_range)
    def get_amount_range(self):
        self.amount_range = self.amountwidget.getValue()
        self.amountChanged.emit(self.amount_range)
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
    def getStrategy(self):
        return self.strategy
    def updateHistogram(self,hist):
        self.histogramWidget.update_data(hist)
        self.energywidget.setRange((hist[:,0].min(),hist[:,0].max()))
        self.amountwidget.setRange((hist[:,1].min(),hist[:,1].max()))
        print(hist[:,1].min())
        print(hist[:,1].max())

   
    def updateLayers(self):
        layers = len(helpers.get_arrow_files(self.arrow_folder.absolutePath()))
        self.layerwidget.setRange((1,layers))
        lowerbound = layers - 10 if layers - 10 >= 1 else 1 
        self.layerwidget.setValue((lowerbound,layers))
        

    def filterHistogram(self):
        self.histoFilter.start_loading()
        files = helpers.get_arrow_files(self.arrow_folder.absolutePath())
        task = helpers.HistogramFilterTask(self.channel,natsorted(files))
        task.signals.filteredHistogram.connect(self.histoFilter.stop_loading)
        self.histopool.start(task)

    def beginRecalculation(self):
        self.layer = self.layerwidget.getValue()
        if self.wav_folder.exists():
            wav_folder = helpers.get_wav_files(self.wav_folder.absolutePath())
            if self.layer[1] < len(wav_folder):
                self.layer_display.setText(str(wav_folder[self.layer[0]].name) + ", "  + str(wav_folder[self.layer[1]].name))
        self.resolution = self.resolutionwidget.value()
        self.channel = self.channelwidget.currentText()
        self.strategy = self.aggregationWidget.currentText()
        self.begincalculation.emit()
    def startCalculation(self):
        self.recalculate.start_loading()
    def finishCalculation(self):
        self.recalculate.stop_loading()

