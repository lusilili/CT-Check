import json
import random
import sys
import time


from PySide6 import QtCharts
from PySide6.QtGui import QPixmap, Qt, QColor, QPen, QFont
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import *
from PySide6 import *
from PySide6.QtCore import Slot, QUrl
from pyecharts.faker import Faker
from topnav import Topnav
from pyecharts.options import InitOpts, LabelOpts, LegendOpts, TooltipOpts

from showct import Ctwid
from check_history import C_History
from bottombar import Bottombar
import pyecharts.options as opts
from pyecharts.charts import Pie, Bar


from model import convnext_tiny as create_model
import torch
from PIL import Image
from torchvision import transforms

# json_path = './class_indices.json'
# assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
#
# with open(json_path, "r") as f:
#     class_indict = json.load(f)

class_indict = {         # 记录标签和数字的关系
    "0": "新冠肺炎",
    "1": "正常肺部",
    "2": "病毒性肺炎"
}
num_classes = 3
img_size = 224
data_transform = transforms.Compose(
    [transforms.Resize(int(img_size * 1.14)),
     transforms.CenterCrop(img_size),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

# create model
model = create_model(num_classes=num_classes).to(device)
# load model weights
model_weight_path = "weights/convnext_tiny_1k_224_ema.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()
class MainWindow(QMainWindow):
    def __init__(self,label,num):
        super().__init__()
        self.setWindowTitle("CT Pie Chart")
        self.resize(400, 300)

        self.chart_view = QtCharts.QChartView(self)

        self.setCentralWidget(self.chart_view)
        self.chart = QtCharts.QChart()
        legend = self.chart.legend()
        # legend.setFont(QFont("Arial", 12))  # 设置图例字体
        # legend.setColor(QColor("red"))  # 设置图例颜色
        legend.setLabelColor(QColor("white"))
        self.chart.setBackgroundBrush(QColor(20, 35, 56))
        self.chart.setTitle("CT Pie Chart")
        self.chart.setTitleBrush(QColor(Qt.white))

        # self.chart.setTitleBrush()
        self.chart_view.setChart(self.chart)
        self.series = QtCharts.QPieSeries()
        self.chart.addSeries(self.series)

        self.update_data(label,num)  # 绘制时更新一次数据

    def update_data(self,label1,num):
        # 更新饼状图数据
        # num=0.5
        print(num)
        label1=[label1,'']
        self.series.clear()
        labels =label1
        sizes = [num,1-num]  # 示例数据，您可以根据需要更改
        total = 1
        for label, size in zip(labels, sizes):
            slice_item = self.series.append(label, size)
            slice_item.setLabelVisible(True)
            # # 在每个饼状图切片中心显示百分比
            percent_text = f"{size*100:.2f}%"
            slice_item.setLabelColor(QColor(Qt.white))
            slice_item.setLabel(f"{label}: {percent_text}")
        # font = QFont()
        # font.setBold(True)  # 设置为粗体
        # font.setPointSize(12)  # 设置字体大小
        # slice_item.setLabelFont(font)

        # print(f"{label}: {percent_text}")
        # 设置第二个切片QPieSlice的格式
        self.slice = self.series.slices()[0]
        # 使其分离饼图中心 (该方法的形参exploded默认为True)
        self.slice.setExploded()
        # 使标签可见 (该方法的形参visible默认为True)
        self.slice.setLabelVisible()
        self.series.slices()[1].setLabelVisible(False)

        # self.slice.setLabelBrush(QColor(Qt.red))



class Rightcontent(QWidget):
    def __init__(self):
        super().__init__()
        
        
        #右侧最大布局
        self.rightcontent_layout = QVBoxLayout()
        
        
        #导入的顶部导航栏
        self.topnav_group = Topnav()
        self.rightcontent_layout.addWidget(self.topnav_group.topnav_group)
        #self.rightcontent_layout.addStretch()

        # #堆叠布局
        self.rightstack_layout = QStackedLayout()
        # 对堆叠布局数据
        self.rightcontentdata = [{"title": "选择图像"},
                                 {"title": "检测图像"},
                                 {"title": "导出结果"},
                                 {"title": "检测记录"}]

        # ####堆叠布局之工作台
        self.stacklayout_work = Ctwid("/home/yuan/Pictures/6e335d2ef0ed37fb4595d3743c20e3f3.jpeg")

        #####堆叠布局之表格主体
        self.stacklayout_history = C_History()

        ####堆叠布局之组件预览
        # self.stacklayout_output = Ct_output()
        #
        #
        # ####堆叠布局之后台管理
        self.stacklayout_management = QWidget()

        # 堆叠布局导入四个tab
        self.rightstack_layout.addWidget(self.stacklayout_work.layout_QV_group)
        self.rightstack_layout.addWidget( self.stacklayout_history.history_group)
        # self.rightstack_layout.addWidget(self.stacklayout_widgetview.layout_QV_group)
        # self.rightstack_layout.addWidget(self.stacklayout_management.layout_QV_group)

        # 堆叠布局纺放入大布局里
        self.rightcontent_layout.addLayout(self.rightstack_layout)







        #导入右侧bottombar
        self.bottombar_group = Bottombar()
        self.rightcontent_layout.addWidget(self.bottombar_group.bottombar_group)
        # self.rightcontent_layout.setStretch(2,5)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*);;Text Files (*.txt)",
                                                   options=options)
        if file_name:
            print("选择的文件路径：", file_name)
            widget_to_remove = self.rightstack_layout.widget(0)
            self.img_path=file_name
            self.rightstack_layout.removeWidget(widget_to_remove)
            self.rightstack_layout.insertWidget(0,Ctwid(file_name).layout_QV_group)
            self.stacklayout_history.edit.appendPlainText('\n')

            self.stacklayout_history.edit.appendHtml(f'<span style=color:white>1{file_name}</span>')
            # self.stacklayout_work.pic_path=file_name
    def save_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*);;Text Files (*.txt)",
                                                   options=options)
        if file_name:
            print("Selected file:", file_name)
            print(self.stacklayout_history.edit.toPlainText())
            with open(file_name, "w") as f:
                f.write(self.stacklayout_history.edit.toPlainText())
    def show_chart(self,label,num):
        # 创建图表对话框
        # m=MyWindow()
        # m.show()
        self.chart_dialog = MainWindow(label,num)
        # self.chart_dialog.draw_chart()
        self.chart_dialog.show()
        #
    def shibie(self):
        try:
            img = Image.open(self.img_path).convert("RGB")
            image=img.copy()
            # plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = "{}  概率: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())

            self.stacklayout_history.edit.appendHtml(f'<span style=color:white>1{print_res}</span>')
            print(print_res)
            self.show_chart(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
        except Exception as e:
            print(e)

    
    #接受leftmenu发出的信号方法
    @Slot(str)
    def getmenuindex(self, msg):
        if msg["index"] == 3:
            self.rightstack_layout.setCurrentIndex(1)
        # print(msg['index'])
        elif msg["index"] == 0:

            self.open_file_dialog()
            self.rightstack_layout.setCurrentIndex(0)
        elif msg["index"] == 2:
            # self.rightstack_layout.setCurrentIndex(0)
            self.save_file_dialog()
        elif msg["index"] == 1:
            self.shibie()

        self.rightstack_layout.update