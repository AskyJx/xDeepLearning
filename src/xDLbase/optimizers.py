"""
Created: May 2018
@author: JerryX
Find more : https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np

# 自适应lr优化类,传入参数元组和梯度元组，完成更新
class AdagradOptimizer(object):
    def __init__(self, optmParams, dataType):

        self.eps = optmParams
        self.dataType = dataType
        self.isInited = False
        self.g=[]

    # lazy init
    def initG(self, w):
        if (False == self.isInited):
            for i in range(len(w)):
                self.g.append( np.zeros(w[i].shape, dtype=self.dataType))
            self.isInited = True

    # w和dw都是元组类型
    def getUpdWeights(self, w, dw, lr):
        self.initG(w)
        wNew = []
        for i in range(len(w)):
            wi, self.g[i] = self.OptimzAdagrad(w[i], dw[i], self.g[i], lr)
            wNew.append(wi)

        # 转为元组输出
        return tuple(wNew)

    def OptimzAdagrad(self, x, dx, g,lr):

        g += dx ** 2
        x += - lr * dx / (np.sqrt(g) + self.eps)

        return x, g

# 自适应矩估计优化类
class AdamOptimizer(object):
    def __init__(self, optmParams, dataType):
        self.beta1 ,self.beta2 , self.eps = optmParams
        self.dataType = dataType
        self.isInited = False
        self.m=[]
        self.v=[]
        # self.m_w = []
        # self.v_w = []
        # self.m_b = []
        # self.v_b = []
        self.Iter = 0

    # lazy init
    def initMV(self, w):
        if (False == self.isInited):
            for i in range(len(w)):
                self.m.append(np.zeros(w[i].shape, dtype=self.dataType))
                self.v.append(np.zeros(w[i].shape, dtype=self.dataType))

            self.isInited = True

    def getUpdWeights(self, w, dw, lr):
        self.initMV(w)

        t = self.Iter + 1
        wNew = []
        for i in range(len(w)):
            wi, self.m[i],self.v[i] = self.OptimzAdam(w[i], dw[i], self.m[i], self.v[i], lr, t)
            wNew.append(wi)

        return tuple(wNew)

    def OptimzAdam(self, x, dx, m, v, lr, t):
        m = self.beta1 * m + (1 - self.beta1) * dx
        mt = m / (1 - self.beta1 ** t)
        v = self.beta2 * v + (1 - self.beta2) * (dx ** 2)
        vt = v / (1 - self.beta2 ** t)
        x += - lr * mt / (np.sqrt(vt) + self.eps)

        return x, m, v
