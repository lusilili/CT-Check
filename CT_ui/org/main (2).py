import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import convnext_tiny as create_model
# from model import convnext_small as create_model
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib as mpl
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from win1 import Ui_MainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import openpyxl
import cv2
import sys
mpl.use('TkAgg')  # !IMPORTANT 更改在这里！！！！！！！！！

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

num_classes = 3
img_size = 224
data_transform = transforms.Compose(
    [transforms.Resize(int(img_size * 1.14)),
     transforms.CenterCrop(img_size),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r") as f:
    class_indict = json.load(f)

class_indict = {         # 记录标签和数字的关系
    "0": "新冠肺炎",
    "1": "正常肺部",
    "2": "病毒性肺炎"
}
# create model
model = create_model(num_classes=num_classes).to(device)
# load model weights
model_weight_path = "weights/convnext_tiny_1k_224_ema.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(Qt.CustomizeWindowHint)

        self.minButton_2.clicked.connect(self.showMinimized)#缩小
        self.maxButton_2.clicked.connect(self.max_or_restore)#放大
        self.closeButton_2.clicked.connect(self.close)#关闭
        self.pushButton.clicked.connect(self.open_file)  # 文件按钮
        self.pushButton_2.clicked.connect(self.shibie)  # 文件按钮
        self.pushButton_3.clicked.connect(self.excel)  # 文件按钮
        self.img_path = "input_data/COVID/COVID (1).png"
        self.zl1_2.setPlaceholderText('请选择需要检测的图片，然后点击检测按钮')
        self.txt="guhanyu"
    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src = cv2.resize(img_src, (nw, nh))
            print(type(img_src))
            frame = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)#转化为qt格式
            print(type(img))
            label.setPixmap(QPixmap.fromImage(img))#QPixmap.fromImage(img)：将Qimage图片转化为QPixmap
        except Exception as e:
            print(repr(e))

    def open_file(self):
        # 获取打开的文件路径
        try:
            name, _ = QFileDialog.getOpenFileName(self, '选取图片')
            print(name)
            if name:
                self.img_path = name
                self.zl1_2.setPlaceholderText("选择文件为："+self.img_path)
                self.listWidget.addItem("选择文件:"+self.img_path)  # 将最后的结果进行展示

                img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

                print(type(img))
                self.show_image(img,self.label_7)

            else:
                self.zl1_2.setPlaceholderText("未选中文件")
        except Exception as e:
            self.statistic_msg('%s' % e)
    def statistic_msg(self, msg):
        self.zl1_2.setPlaceholderText(msg)              #连接行编辑控件
    def shibie(self):
        try:
            img = Image.open(self.img_path).convert("RGB")
            image=img.copy()
            plt.imshow(img)
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
            self.zl1_2.setPlaceholderText("检测结果为：" + str(print_res))
            self.listWidget.addItem("检测结果为：" + str(print_res))  # 将最后的结果进行展示
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("检测结果为：" + str(print_res))
            msg.setWindowTitle("检测结果")
            msg.exec_()
        except Exception as e:
            print(e)
            self.statistic_msg('%s' % e)

    def img_file(self):
        try:
            fileName = QFileDialog.getExistingDirectory(self, "选取文件夹")
            if fileName:
                self.img_path=fileName
                self.zl1_2.setPlaceholderText(self.img_path)
        except Exception as e:
            self.statistic_msg('%s' % e)

    def cv2AddChineseText(self,img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text(position, text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    def opentxt_file(self):
        fileName = QFileDialog.getExistingDirectory(self, "选取文件夹")
        if fileName:
            self.txt = fileName
            self.zl1_2.setPlaceholderText('保存文件地址为：' + fileName)

    def excel(self):
        try:
            self.opentxt_file()
            widgetres = []
            filename = self.txt
            filename = filename + '/检测记录.xlsx'
            wb = openpyxl.Workbook()
            ws = wb['Sheet']
            # 获取listwidget中条目数
            count = self.listWidget.count()
            # 遍历listwidget中的内容
            for i in range(count):
                widgetres.append(self.listWidget.item(i).text())
            print(widgetres)

            for i, item in enumerate(widgetres, 1):
                ws.cell(row=i, column=1).value = item
            wb.save(filename)
        except Exception as e:
            self.statistic_msg('%s' % e)


    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True
            else:
                self.m_flag = False

    def max_or_restore(self):#界面放大
        if self.maxButton_2.isChecked() and self.m_flag:
            self.showMaximized()
        else:
            self.showNormal()

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        if self.maxButton_2.isChecked():
            self.setCursor(Qt.ArrowCursor)  # 恢复正常鼠标状态

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
