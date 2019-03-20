"""
Created: May 2018
@author: JerryX
Find more : https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np
import numba

class Tools:
    # padding before cross-correlation and pooling
    @staticmethod
    def padding(x, pad, data_type):

        size_x = x.shape[2]  # 输入矩阵尺寸
        size = size_x + pad * 2  # padding后尺寸
        if x.ndim == 4:  # 每个元素是3维的，x的0维是mini-batch
            # 初始化同维全0矩阵
            padding = np.zeros((x.shape[0], x.shape[1], size, size), dtype=data_type)
            # 中间以x填充
            padding[:, :, pad: pad + size_x, pad: pad + size_x] = x

        elif x.ndim == 3:  # 每个元素是2维的
            padding = np.zeros((x.shape[0], size, size), dtype=data_type)
            padding[:, pad: pad + size_x, pad: pad + size_x] = x

        return padding

    # 执行环境内存充裕blas方法较快
    # 否则使用jit后的np.matmul方法
    @numba.jit
    def matmul(a, b):
        return np.matmul(a, b)

    # 输出层结果转换为标准化概率分布，
    # 入参为原始线性模型输出y ，N*K矩阵，
    # 输出矩阵规格不变
    @staticmethod
    def softmax(y):
        # 对每一行：所有元素减去该行的最大的元素,避免exp溢出,得到1*N矩阵,
        max_y = np.max(y, axis=1)
        # 极大值重构为N * 1 数组
        max_y.shape = (-1, 1)
        # 每列都减去该列最大值
        y1 = y - max_y
        # 计算exp
        exp_y = np.exp(y1)
        # 按行求和，得1*N 累加和数组
        sigma_y = np.sum(exp_y, axis=1)
        # 累加和reshape为N*1 数组
        sigma_y.shape = (-1, 1)
        # 计算softmax得到N*K矩阵
        softmax_y = exp_y / sigma_y

        return softmax_y

    # 交叉熵损失函数
    # 限制上界避免除零错
    @staticmethod
    def crossEntropy(y,y_):
        return -np.log(np.clip(y[range(len(y)), y_],1e-10,None,None))

    # 平方误差损失
    @staticmethod
    def mse(y, y_):
        return np.mean((y - y_) ** 2, axis=1) / 2

    # sigmoid
    @staticmethod
    def sigmoid(x):

        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = np.zeros_like(x)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[neg_mask] = np.exp(x[neg_mask])
        top = np.ones_like(x)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)
