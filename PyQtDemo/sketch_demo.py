import sys
import numpy as np
import onnxruntime as ort
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton
from PyQt6.QtGui import QPainter, QPen, QImage, QColor
from PyQt6.QtCore import Qt, QPoint, QTimer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

CLASSES = ["bicycle", "eyeglasses", "car", "eye", "tree", "apple", "smiley_face", "cell_phone", "airplane", "book"]

class DrawingWidget(QWidget):
    def __init__(self, update_callback):
        super().__init__()
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.black)  # black background
        
        self.drawing = False
        self.last_point = QPoint()
        self.update_callback = update_callback
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.trigger_update)
        self.timer.setSingleShot(True)
        
    def clear(self):
        self.image.fill(Qt.GlobalColor.black)
        self.update()
        self.update_callback(self.get_image_array())
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.GlobalColor.white, 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()
            
            # Reset timer to update prediction shortly after the stroke
            self.timer.start(50) 
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.trigger_update()
            
    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())
        
    def trigger_update(self):
        img_array = self.get_image_array()
        self.update_callback(img_array)
        
    def get_image_array(self):
        # Scale to 28x28 for ONNX model input
        scaled_img = self.image.scaled(28, 28, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        gray = np.zeros((28, 28), dtype=np.float32)
        for y in range(28):
            for x in range(28):
                gray[y, x] = scaled_img.pixelColor(x, y).red() / 255.0
        
        # Expand dims for batch and channel: (batch=1, channel=1, height=28, width=28)
        gray = np.expand_dims(gray, axis=(0, 1))
        print(gray.shape)
        print(gray)
        return gray

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EmLogic Sketch Recognition Demo")
        
        # Load ONNX model
        model_path = "../Checkpoints/model_8bit_qcdq_clean.onnx"
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Central widget
        central = QWidget()
        layout = QHBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)
        
        # Left side layout
        left_layout = QVBoxLayout()
        self.drawing_widget = DrawingWidget(self.on_draw_update)
        left_layout.addWidget(self.drawing_widget)
        
        self.btn_clear = QPushButton("Clear Canvas")
        self.btn_clear.clicked.connect(self.drawing_widget.clear)
        left_layout.addWidget(self.btn_clear)
        left_layout.addStretch()
        
        layout.addLayout(left_layout)
        
        # Right side layout (Matplotlib)
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.init_plot()
        self.drawing_widget.clear() # trigger initial plot 

    def init_plot(self):
        self.ax.clear()
        self.bars = self.ax.barh(CLASSES, np.zeros(10), color='skyblue')
        self.ax.set_xlim(0, 1)
        self.ax.set_xlabel('Probability')
        self.ax.set_title('Live Prediction')
        self.ax.invert_yaxis()  # to match the order in CLASSES from top to bottom
        self.figure.tight_layout()
        self.canvas.draw()
        
    def on_draw_update(self, img_array):
        try:
            ort_inputs = {self.input_name: img_array}
            ort_outs = self.session.run(None, ort_inputs)
            logits = ort_outs[0][0]
            
            # Apply Softmax to get probabilities
            print(CLASSES)
            print(logits)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
        except Exception as e:
            print(f"Inference error: {e}")
            probs = np.zeros(10)
            
        # Update plot bars
        for bar, prob in zip(self.bars, probs):
            bar.set_width(prob)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
