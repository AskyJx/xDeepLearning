"""
Created: May 2018
@author: JerryX
Find more : https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 训练完成后，生成训练过程图像
class ResultView(object):
    def __init__(self, epoch, line_labels, colors, ax_labels, dataType):
        self.cur_p_idx = 0
        self.curv_x = np.zeros(epoch * 100, dtype=int)
        self.curv_ys = np.zeros((4, epoch * 100), dtype=dataType)
        self.line_labels = line_labels
        self.colors = colors
        self.ax_labels = ax_labels

    def addData(self, curv_x, loss, loss_v, acc, acc_v):

        self.curv_x[self.cur_p_idx] = curv_x
        self.curv_ys[0][self.cur_p_idx] = loss
        self.curv_ys[1][self.cur_p_idx] = loss_v
        self.curv_ys[2][self.cur_p_idx] = acc
        self.curv_ys[3][self.cur_p_idx] = acc_v
        self.cur_p_idx += 1

    # 显示曲线
    def show(self):
        self.showCurves(self.cur_p_idx, self.curv_x, self.curv_ys, self.line_labels, self.colors, self.ax_labels)

    def showCurves(self, idx, x, ys, line_labels, colors, ax_labels):
        LINEWIDTH = 2.0
        plt.figure(figsize=(8, 4))
        # loss
        ax1 = plt.subplot(211)
        for i in range(2):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], linewidth=LINEWIDTH, label=line_labels[i])

        ax1.xaxis.set_major_locator(MultipleLocator(4000))
        ax1.yaxis.set_major_locator(MultipleLocator(0.1))
        ax1.set_xlabel(ax_labels[0])
        ax1.set_ylabel(ax_labels[1])
        plt.grid()
        plt.legend()

        # Acc
        ax2 = plt.subplot(212)
        for i in range(2, 4):
            line = plt.plot(x[:idx], ys[i][:idx])[0]
            plt.setp(line, color=colors[i], linewidth=LINEWIDTH, label=line_labels[i])

        ax2.xaxis.set_major_locator(MultipleLocator(4000))
        ax2.yaxis.set_major_locator(MultipleLocator(0.02))
        ax2.set_xlabel(ax_labels[0])
        ax2.set_ylabel(ax_labels[2])

        plt.grid()
        plt.legend()
        plt.show()
