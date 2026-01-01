
import helperfunctions as helpers
import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication, QHBoxLayout,QVBoxLayout, QWidget, QLabel, QPushButton

from PySide6.QtOpenGLWidgets import QOpenGLWidget
import numpy as np
import ctypes
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
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, glGetString,GL_VERSION,GL_PROGRAM_POINT_SIZE, glEnable, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER
)




VERTEX_SHADER = """
#version 330 core

layout (location = 0) in vec2 in_pos;
layout (location = 1) in float in_value;

uniform mat4 u_transform;
uniform float u_pointSize;

out float v_value;

void main()
{
    gl_Position = u_transform * vec4(in_pos, 0.0, 1.0);
    gl_PointSize = u_pointSize;
    v_value = in_value;

// Bypass the transform matrix for a moment
//    gl_Position = vec4(in_pos, 0.0, 1.0);
//    gl_PointSize = u_pointSize;
//    v_value = in_value;
}
"""
FRAGMENT_SHADER = """
#version 330 core

in float v_value;

uniform float vmin;
uniform float vmax;
uniform sampler1D colormap;

out vec4 fragColor;

void main()
{
    float t = clamp((v_value - vmin) / (vmax - vmin), 0.0, 1.0);
    fragColor = texture(colormap, t);
}
"""




def viridis_colormap(n=256):
    # compact viridis approximation (no matplotlib dependency)
    cmap = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        t = i / (n - 1)
        cmap[i] = [
            0.5 + 0.5 * np.cos(3.0 + t * 5.0),
            0.5 + 0.5 * np.cos(1.5 + t * 5.0),
            0.5 + 0.5 * np.cos(0.0 + t * 5.0),
            1.0
        ]
    return cmap



class PointCloud2D(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None
        self.point_count = 0

        # view state
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pos = None

        # rendering options
        self.point_size = 1.0
        self.resolution = 10
        self.vmin = 0.0
        self.vmax = 32767.0
        self.program = 0
        self.u_transform = None
        self.vao = 0
        self.vbo = 0

    # ---------- public API ----------

    def set_points(self, data: np.ndarray):
        """data shape: (N, 3) -> x, y, value"""
        self.data = data.astype(np.float32)
        self.point_count = len(data)

        if self.isValid():
            self._upload_data()

        self.update()

    def set_point_size(self, size: float):
        self.point_size = float(size)
        self.update()

    def set_value_range(self, value_range):
        self.vmin = float(value_range[0])
        self.vmax = float(value_range[1])
        self.update()

    # ---------- Qt / OpenGL ----------

    def initializeGL(self):
        glEnable(GL_PROGRAM_POINT_SIZE)
        self.makeCurrent()
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        # CRITICAL: If these are 0, the driver failed to provide a buffer

        glClearColor(0, 0, 0, 1)
        self.program = self._create_program()
        self._get_uniforms()

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        self._create_colormap()

        if self.isValid():
            self._upload_data()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)


        if self.program is None or self.point_count == 0 or self.u_transform is None or self.vao is None or self.vbo is None:
            return


        glUseProgram(self.program)
        glBindVertexArray(self.vao)
        transform = self._make_transform()
        glUniformMatrix4fv(self.u_transform, 1, GL_FALSE, transform)
        glUniform1f(self.u_pointSize, self.point_size)
        glUniform1f(self.u_vmin, self.vmin)
        glUniform1f(self.u_vmax, self.vmax)

        glDrawArrays(GL_POINTS, 0, self.point_count)

    # ---------- mouse interaction ----------

#    def wheelEvent(self, event):
#        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
#        self.zoom *= factor
#        self.update()

    def wheelEvent(self, event):
        # 1. Get mouse position in widget pixels
        pos = event.position()

        # 2. Convert pixel coordinates to Normalized Device Coordinates (-1 to 1)
        # OpenGL Y is inverted compared to Qt pixels
        ndc_x = (2.0 * pos.x() / self.width()) - 1.0
        ndc_y = 1.0 - (2.0 * pos.y() / self.height())

        # 3. Determine zoom factor
        zoom_step = 1.1 if event.angleDelta().y() > 0 else 0.9
        old_zoom = self.zoom
        self.zoom *= zoom_step

        # 4. Adjust pan so the NDC point stays under the cursor
        # Formula: NewPan = NDC - (NDC - OldPan) * (NewZoom / OldZoom)
        self.pan_x = ndc_x - (ndc_x - self.pan_x) * zoom_step
        self.pan_y = ndc_y - (ndc_y - self.pan_y) * zoom_step

        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return

        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()

        self.pan_x += 2.0 * dx / self.width()
        self.pan_y -= 2.0 * dy / self.height()

        self.last_pos = event.position()
        self.update()

    def mouseReleaseEvent(self, event):
        self.last_pos = None

    # ---------- internal helpers ----------
    def _make_transform(self):
        # This is a standard 4x4 Identity matrix modified for 2D pan/zoom
        # We use Column-Major layout here so we can use GL_FALSE
        return np.array([
            [self.zoom, 0,         0, 0],
            [0,         self.zoom, 0, 0],
            [0,         0,         1, 0],
            [self.pan_x, self.pan_y, 0, 1]
        ], dtype=np.float32)

    def _upload_data(self):
        self.makeCurrent()
        glBindVertexArray(self.vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # Ensure data is 32-bit floats
        glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, self.data, GL_STATIC_DRAW)

        # Stride is 12 bytes: [x(4), y(4), val(4)]
        # Location 0: x, y
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Location 1: value
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _create_colormap(self):
        self.cmap_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.cmap_tex)

        cmap = viridis_colormap(256)

        glTexImage1D(
            GL_TEXTURE_1D, 0, GL_RGBA32F,
            256, 0, GL_RGBA, GL_FLOAT, cmap
        )

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "colormap"), 0)

    def _create_program(self):

        vs = helpers.compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fs = helpers.compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)

        if not glGetProgramiv(prog, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(prog).decode())

        glDeleteShader(vs)
        glDeleteShader(fs)

        return prog

    def _get_uniforms(self):
        self.u_transform = glGetUniformLocation(self.program, "u_transform")
        self.u_pointSize = glGetUniformLocation(self.program, "u_pointSize")
        self.u_vmin = glGetUniformLocation(self.program, "vmin")
        self.u_vmax = glGetUniformLocation(self.program, "vmax")
    def __del__(self):
        # This "empty" destructor prevents PyOpenGL from 
        # trying to call glDelete* during Python shutdown.
        pass




