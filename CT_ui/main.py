import sys
from PySide6 import *
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from pyecharts.charts import Pie
from pyecharts.options import InitOpts, LabelOpts, LegendOpts, TooltipOpts
import pyecharts.options as opts
from mainwindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    

    window = MainWindow()

    window.resize(1920/2,1080/2)
    window.show()
    
    
    sys.exit(app.exec())