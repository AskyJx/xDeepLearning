"""
Created: May 2018
@author: JerryX
Find more : https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from xDLUtils import Tools


# 全连接类
class FCLayer(object):
    def __init__(self, miniBatchesSize, i_size, o_size,
                 activator, optimizerCls,optmParams,
                 dataType,init_w):
        # 初始化超参数
        self.miniBatchesSize = miniBatchesSize
        self.i_size = i_size
        self.o_size = o_size
        self.activator = activator
        self.optimizerObj = optimizerCls(optmParams, dataType)
        self.dataType = dataType
        self.w = init_w * np.random.randn(i_size, o_size).astype(dataType)
        self.b = np.zeros(o_size, dataType)
        self.out = []
        self.deltaPrev = []  
        self.deltaOri = []  

    # 预测时前向传播
    def inference(self, input):
        self.out = self.fp(input)
        return self.out

    # 前向传播,激活后再输出
    def fp(self, input):
        self.out = self.activator.activate(Tools.matmul(input, self.w) + self.b)
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta, lrt):
        self.deltaOri = self.activator.bp(delta, self.out)

        self.bpDelta()
        self.bpWeights(input, lrt)

        return self.deltaPrev

    # 输出误差反向传播至上一层
    def bpDelta(self):
        self.deltaPrev = Tools.matmul(self.deltaOri, self.w.T)
        return self.deltaPrev

    # 计算反向传播权重梯度w,b
    def bpWeights(self, input, lrt):
        dw = Tools.matmul(input.T, self.deltaOri)
        db = np.sum(self.deltaOri, axis=0, keepdims=True).reshape(self.b.shape)
        weight = (self.w,self.b)
        dweight = (dw,db)
        # 元组按引用传递
        self.optimizerObj.getUpdWeights(weight,dweight, lrt)