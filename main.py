# Copyright (C) 2013 Riverbank Computing Limited.
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
from __future__ import annotations


import sys
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication,QSlider, QHBoxLayout,QVBoxLayout, QWidget, QLabel, QPushButton, QSpinBox, QComboBox, QFileDialog

from PySide6.QtOpenGLWidgets import QOpenGLWidget

from superqt import QRangeSlider
from pathlib import Path
import helperfunctions as helpers
import numpy as np
import openglwidget as glw
import sidebar as sidebar



class Window(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        mainLayout = QHBoxLayout(self)
        self.active_threads = []
        self.sidebar = sidebar.Sidebar()
        self.sidebar.setMinimumSize(200,700)
        self.glwidget = glw.PointCloud2D(parent=self)
        self.glwidget.setMinimumSize(700, 700)
        self.glwidget.set_point_size(3.0)
        self.glwidget.set_value_range((0,2**15))

        mainLayout.addWidget(self.glwidget)
        mainLayout.addWidget(self.sidebar)
        
        '''Draw Connections'''
        self.sidebar.energyChanged.connect(self.glwidget.set_value_range)
        self.sidebar.pointsizeChanged.connect(self.glwidget.set_point_size)
        '''Calculation Connections'''
        self.sidebar.begincalculation.connect(self.handle_array_update)
        self.sidebar.export.connect(self.export)
        self.setWindowTitle(self.tr("Ebm Visualisation"))

    def handle_array_update(self):
        nth = self.sidebar.getResolution()
        ch = self.sidebar.getChannel()
        layer = self.sidebar.getLayer()
        folder = self.sidebar.getCsvFolder()

        thread = QThread()
        arrow_files = helpers.get_arrow_files(folder.absolutePath())
        worker = helpers.DataWorker(nth, ch, arrow_files[layer-1])
        worker.moveToThread(thread)
        self.active_threads.append(thread)
        thread.worker = worker
        thread.started.connect(worker.run)
        thread.worker.data_ready.connect(self.on_data_received)
        thread.worker.finished.connect(thread.quit) # Stop thread when done
        thread.worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.sidebar.startCalculation()
        thread.finished.connect(lambda: self.active_threads.remove(thread))
        thread.start()

    def on_data_received(self, arr):
        self.glwidget.set_points(arr)
        self.sidebar.finishCalculation()

    def export(self):
        image = self.glwidget.grabFramebuffer()
        
        success = image.save("output_capture.png", "PNG")
        
        if success:
            print("Export successful!")
        else:
            print("Export failed.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = Window()
    window.show()
    sys.exit(app.exec())
