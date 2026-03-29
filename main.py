# Copyright (C) 2013 Riverbank Computing Limited.
# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
from __future__ import annotations


import sys
from PySide6.QtCore import Qt, Signal, QThread, QThreadPool
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication,QSlider, QHBoxLayout,QVBoxLayout, QWidget, QLabel, QPushButton, QSpinBox, QComboBox, QFileDialog, QTabWidget, QTextEdit

from PySide6.QtOpenGLWidgets import QOpenGLWidget

from superqt import QRangeSlider
from pathlib import Path
import helperfunctions as helpers
import numpy as np
import openglwidget as glw
import sidebar as sidebar
import faulthandler

# Enable faulthandler for segfaults, even in GUI apps
#faulthandler.enable(file=sys.stderr, all_threads=True)


class VisualizerTab(QWidget):

    def __init__(self, parent=None):
        super().__init__()

        mainLayout = QHBoxLayout(self)
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
        folder = self.sidebar.getArrowFolder()

        arrow_files = helpers.get_arrow_files(folder.absolutePath())
        if layer[1]-1 not in range(len(arrow_files)):
            return
            
        #worker = helpers.DataWorker(nth, ch, [arrow_files[layer[1]-1]])

        #print(layer)
        if layer[0] == layer[1]:
            worker = helpers.DataWorker(nth, ch, [arrow_files[layer[0]-1]])
        else :
            worker = helpers.DataWorker(nth, ch, arrow_files[layer[0]-1:layer[1]-1])

        self.pool = QThreadPool.globalInstance()
        worker.carrier.finished.connect(self.on_data_received)
        self.pool.start(worker)

        self.sidebar.startCalculation()

    def on_data_received(self, arr):
        arr = np.ascontiguousarray(arr)
        self.glwidget.set_points(arr)
        self.sidebar.finishCalculation()

    def export(self):
        image = self.glwidget.grabFramebuffer()
        
        success = image.save("output_capture.png", "PNG")
        
        if success:
            print("Export successful!")
        else:
            print("Export failed.")

    def closeEvent(self, event):
            """
            Wait for all threads in the global pool to finish 
            before allowing the window to close.
            """
            pool = QThreadPool.globalInstance()
            
            # This tells the pool not to start any NEW tasks
            # and waits for current ones to finish.
            pool.waitForDone() 
            
            event.accept()

class ObpcreatorTab(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        layout = QHBoxLayout(self)
        editor = QTextEdit()
        layout.addWidget(editor)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    
    visualizerTab = VisualizerTab()
    visualizerTab.show()
    sys.exit(app.exec())




