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

# Rnn类
class RnnLayer(object):

    def __init__(self,LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False # 初始化标志
        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        self.rnnParams = []
        self.deltaPrev = []  

    def initNnWeight(self, D, H, layersNum, dataType):

        #层次
        rnnParams = []
        for layer in range(layersNum):

            Wh = np.random.uniform(-1*self.init_rng, self.init_rng,(H, H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1*self.init_rng, self.init_rng,(D, H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1*self.init_rng, self.init_rng,(H, H)).astype(dataType)
            b = np.zeros(H, dataType)
            rnnParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})

        self.rnnParams = rnnParams


    # 预测时前向传播
    def inference(self, input):
        self.out = self.fp(input)
        return self.out

    # 前向传播,激活后再输出
    def fp(self,x):
        N,T,D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self.initNnWeight(D,H,L,self.dataType)
            self.isInited = True


        h = self.rnn_forward(x)
        # N进 v 1出 模型，只保留时序最后的一项输出
        self.out = h[:,-1,:]
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta, lrt):


        N,T,D = input.shape
        H = delta.shape[1]
        dh = np.zeros((N,T,H),self.dataType)
        dh[:,-1,:] = delta
        dx, dweight = self.rnn_backward(dh)

        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L-1,-1,-1):

            w = (self.rnnParams[l]['Wx'], self.rnnParams[l]['Wh'], self.rnnParams[l]['b'])
            self.optimizerObjs[l].getUpdWeights(w,dw[L-1-l],lrt)


    def rnn_forward(self,x):
        """
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D).
        - h0: Initial hidden state, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)

        Returns a tuple of:
        - h: Hidden states for the entire timeseries, of shape (N, T, H).
        - cache: Values needed in the backward pass
        """

        h, cache = None, None
        N,T,D = x.shape
        L = self.layersNum
        H = self.rnnParams[0]['b'].shape[0]
        xh = x
        for layer in range(L):

            h = np.zeros((N,T,H))
            h0 = np.zeros((N,  H))
            cache=[]
            for t in range(T):

                h[:, t, :], tmp_cache = self.rnn_step_forward(xh[:, t, :],
                                                              h[:, t - 1, :] if t > 0 else h0,
                                                              self.rnnParams[layer]['Wx'], self.rnnParams[layer]['Wh'],
                                                              self.rnnParams[layer]['b'])
                cache.append(tmp_cache)
            xh = h 
            self.rnnParams[layer]['h']=h
            self.rnnParams[layer]['cache'] = cache

        return h    


    def rnn_backward(self,dh):
        """
        Inputs:
        - dh: Upstream gradients of all hidden states, of shape (N, T, H).
        Returns a tuple of:
        - dx: Gradient of inputs, of shape (N, T, D)
        - dh0: Gradient of initial hidden state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        - db: Gradient of biases, of shape (H,)
        """
        dx, dh0, dWx, dWh, db = None, None, None, None, None
        ##############################################################################
        # TODO: Implement the backward pass for a vanilla RNN running an entire      #
        # sequence of data. You should use the rnn_step_backward function that you   #
        # defined above. You can use a for loop to help compute the backward pass.   #
        ##############################################################################
        N,T,H = dh.shape
        x, _, _, _, _ = self.rnnParams[0]['cache'][0]
        D = x.shape[1]


        dh_prevl = dh

        dweights=[]
        # 逐层倒推
        for layer in range(self.layersNum-1,-1,-1):
            #得到前向传播保存的cache数组
            cache = self.rnnParams[layer]['cache']

            DH = D if layer == 0 else H
            dx= np.zeros((N,T,DH))
            dWx = np.zeros((DH,H))
            dWh = np.zeros((H,H))
            db = np.zeros(H)
            dprev_h_t = np.zeros((N, H))
            # 倒序遍历
            for t in range(T-1,-1,-1):
                dx[:,t,:], dprev_h_t, dWx_t, dWh_t, db_t = self.rnn_step_backward(dh_prevl[:,t,:]+dprev_h_t,cache[t])
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

            dh_prevl=dx

            dweight = (dWx, dWh, db)
            dweights.append(dweight)

        return dx, dweights

    def rnn_step_forward(self, x, prev_h, Wx, Wh, b):
        """
        Inputs:
        - x: Input data for this timestep, of shape (N, D).
        - prev_h: Hidden state from previous timestep, of shape (N, H)
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases of shape (H,)
        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - cache: Tuple of values needed for the backward pass.
        """

        next_h, cache = None, None
        z = np.matmul(x,Wx)+np.matmul(prev_h,Wh) +b

        next_h = np.tanh(z)

        dtanh = 1. - next_h * next_h
        cache=(x, prev_h, Wx, Wh, dtanh)
        return next_h, cache


    def rnn_step_backward(self, dnext_h, cache):
        """
        Inputs:
        - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        - cache: Cache object from the forward pass

        Returns a tuple of:
        - dx: Gradients of input data, of shape (N, D)
        - dprev_h: Gradients of previous hidden state, of shape (N, H)
        - dWx: Gradients of input-to-hidden weights, of shape (D, H)
        - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        - db: Gradients of bias vector, of shape (H,)
        """
        dx, dprev_h, dWx, dWh, db = None, None, None, None, None

        x, prev_h, Wx, Wh, dtanh = cache
        dz = dnext_h * dtanh
        dx = np.matmul(dz,Wx.T)
        dprev_h = np.matmul(dz,Wh.T)
        dWx = np.matmul(x.T,dz)
        dWh = np.matmul(prev_h.T,dz)
        db = np.sum(dz,axis=0)


        return dx, dprev_h, dWx, dWh, db

# LSTM 类
class LSTMLayer(object):

    def __init__(self,LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams,dataType,init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False # 初始化标志
        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        self.lstmParams = []
        self.deltaPrev = []  

    def initNnWeight(self, D, H, layersNum, dataType):

        #层次
        lstmParams = []
        for layer in range(layersNum):
            Wh = np.random.uniform(-1*self.init_rng, self.init_rng,(H, 4*H)).astype(dataType)
            if (0 == layer):
                Wx = np.random.uniform(-1*self.init_rng, self.init_rng,(D, 4*H)).astype(dataType)
            else:
                Wx = np.random.uniform(-1*self.init_rng, self.init_rng,(H, 4*H)).astype(dataType)
            b = np.zeros(4*H, dataType)

            lstmParams.append({'Wx': Wx, 'Wh': Wh, 'b': b})
        self.lstmParams = lstmParams

    # 预测时前向传播
    def inference(self, input):
        self.out = self.fp(input)
        return self.out

    def fp(self,x):
        N,T,D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self.initNnWeight(D,H,L,self.dataType)
            self.isInited = True

        h = self.lstm_forward(x)
        self.out = h[:,-1,:]
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta, lrt):

        N,T,D = input.shape
        H = delta.shape[1]
        dh = np.zeros((N,T,H),self.dataType)
        dh[:,-1,:] = delta
        dx, dweight = self.lstm_backward(dh)

        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.lstmParams[l]['Wx'], self.lstmParams[l]['Wh'], self.lstmParams[l]['b'])
            self.optimizerObjs[l].getUpdWeights(w, dw[L-1-l], lrt)

    def lstm_forward(self, x):
        """
        Inputs:
        - x: Input data of shape (N, T, D)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        """
        h, cache = None, None
        N,T,D = x.shape
        L = self.layersNum
        H = int(self.lstmParams[0]['b'].shape[0]/4) # 取整
        xh = x  #首层输入是x
        for layer in range(L):
            h = np.zeros((N, T, H))
            h0 = np.zeros((N,H))
            c = np.zeros((N, T, H))
            c0 = np.zeros((N, H))
            cache = []
            for t in range(T):

                h[:, t, :], c[:, t, :], tmp_cache = self.lstm_step_forward(xh[:, t, :], h[:, t - 1, :] if t > 0 else h0,
                                                                      c[:, t - 1, :] if t > 0 else c0,self.lstmParams[layer]['Wx'], self.lstmParams[layer]['Wh'],
                                                              self.lstmParams[layer]['b'])
                cache.append(tmp_cache)
            xh = h # 之后以h作为xh作为跨层输入
            self.lstmParams[layer]['h']=h
            self.lstmParams[layer]['c'] = h
            self.lstmParams[layer]['cache'] = cache
        return h

    def lstm_backward(self, dh):
        """
        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0, dWx, dWh, db = None, None, None, None, None
        N,T,H = dh.shape
        x, _, _, _, _, _, _, _, _, _ = self.lstmParams[0]['cache'][0]
        D = x.shape[1]

        dh_prevl = dh
        dweights=[]

        for layer in range(self.layersNum-1,-1,-1):
            #得到前向传播保存的cache数组
            cache = self.lstmParams[layer]['cache']

            DH = D if layer == 0 else H
            dx = np.zeros((N, T, DH))
            dWx = np.zeros((DH,4*H))

            dWh = np.zeros((H,4*H))
            db = np.zeros((4 * H))
            dprev_h = np.zeros((N,H))
            dprev_c = np.zeros((N,H))
            for t in range(T - 1, -1, -1):
                dx[:, t, :], dprev_h, dprev_c, dWx_t, dWh_t, db_t = self.lstm_step_backward(dh_prevl[:, t, :] + dprev_h, dprev_c,
                                                                                      cache[t])  # 注意此处的叠加
                dWx += dWx_t
                dWh += dWh_t
                db += db_t

            dh_prevl=dx

            dweight = (dWx, dWh, db)
            dweights.append(dweight)

        return dx, dweights

    def lstm_step_forward(self, x, prev_h, prev_c, Wx, Wh, b):
        """
        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h, next_c, cache = None, None, None

        H = prev_h.shape[1]
        #z , of shape(N,4H)
        z = np.matmul(x, Wx) + np.matmul(prev_h, Wh) + b

        # of shape(N,H)
        i = Tools.sigmoid(z[:, :H])
        f = Tools.sigmoid(z[:, H:2 * H])
        o = Tools.sigmoid(z[:, 2 * H:3 * H])
        g = np.tanh(z[:, 3 * H:])
        next_c = f * prev_c + i * g
        next_h = o * np.tanh(next_c)

        cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c)

        return next_h, next_c, cache

    def lstm_step_backward(self, dnext_h, dnext_c, cache):
        """
        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
        x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_c = cache

        dnext_c = dnext_c + o * (1 - np.tanh(next_c) ** 2) * dnext_h  # next_h = o*np.tanh(next_c)
        di = dnext_c * g  # next_c = f*prev_c + i*g
        df = dnext_c * prev_c  # next_c = f*prev_c + i*g
        do = dnext_h * np.tanh(next_c)  # next_h = o*np.tanh(next_c)
        dg = dnext_c * i  # next_h = o*np.tanh(next_c)
        dprev_c = f * dnext_c  # next_c = f*prev_c + i*g
        dz = np.hstack((i * (1 - i) * di, f * (1 - f) * df, o * (1 - o) * do, (1 - g ** 2) * dg))  

        dx = np.matmul(dz, Wx.T)
        dprev_h = np.matmul(dz, Wh.T)
        dWx = np.matmul(x.T, dz)
        dWh = np.matmul(prev_h.T, dz)

        db = np.sum(dz, axis=0)

        return dx, dprev_h, dprev_c, dWx, dWh, db

# GRU 类
class GRULayer(object):

    def __init__(self,LName, miniBatchesSize, nodesNum, layersNum,
                 optimizerCls, optmParams, dataType, init_rng):
        # 初始化超参数
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.nodesNum = nodesNum
        self.layersNum = layersNum
        self.dataType = dataType
        self.init_rng = init_rng
        self.isInited = False # 初始化标志
        self.out = []
        self.optimizerObjs = [optimizerCls(optmParams, dataType) for i in range(layersNum)]
        self.gruParams = []
        self.deltaPrev = []  

    def initNnWeight(self, D, H, layersNum, dataType):

        #层次
        gruParams = []
        for layer in range(layersNum):
            Wzh = np.random.uniform(-1*self.init_rng, self.init_rng,(H, 2*H)).astype(dataType)
            War = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            if (0 == layer):
                Wzx = np.random.uniform(-1*self.init_rng, self.init_rng,(D, 2*H)).astype(dataType)
                Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (D, H)).astype(dataType)
            else:
                Wzx = np.random.uniform(-1*self.init_rng, self.init_rng,(H, 2*H)).astype(dataType)
                Wax = np.random.uniform(-1 * self.init_rng, self.init_rng, (H, H)).astype(dataType)
            bz = np.zeros(2*H, dataType)
            ba = np.zeros( H, dataType)
            gruParams.append({'Wzx': Wzx, 'Wzh': Wzh, 'bz': bz, 'Wax': Wax, 'War': War, 'ba': ba})

        self.gruParams = gruParams

    # 预测时前向传播
    def inference(self, input):
        self.out = self.fp(input)
        return self.out

    def fp(self,x):
        N,T,D = x.shape
        H = self.nodesNum
        L = self.layersNum
        # lazy init
        if (False == self.isInited):
            self.initNnWeight(D,H,L,self.dataType)
            self.isInited = True

        h = self.gru_forward(x)
        self.out = h[:,-1,:]
        return self.out

    # 反向传播方法(误差和权参)
    def bp(self, input, delta, lrt):

        # dw是一个数组，对应结构的多层，每层的dw,dh,db,dh0表示需要参数梯度
        N,T,D = input.shape
        H = delta.shape[1]
        dh = np.zeros((N,T,H),self.dataType)
        dh[:,-1,:] = delta
        dx, dweight = self.gru_backward(dh)

        self.bpWeights(dweight, lrt)

        return dx

    # 计算反向传播权重梯度w,b
    def bpWeights(self, dw, lrt):

        L = self.layersNum
        for l in range(L):
            w = (self.gruParams[l]['Wzx'], self.gruParams[l]['Wzh'], self.gruParams[l]['bz'],self.gruParams[l]['Wax'], self.gruParams[l]['War'], self.gruParams[l]['ba'])
            self.optimizerObjs[l].getUpdWeights(w, dw[L-1-l], lrt)

    def gru_forward(self, x):
        """
        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        h, cache = None, None

        N,T,D = x.shape
        L = self.layersNum
        H = self.gruParams[0]['ba'].shape[0] 
        xh = x  
        for layer in range(L):
            h = np.zeros((N, T, H))
            h0 = np.zeros((N,H))
            cache = []
            for t in range(T):

                h[:, t, :], tmp_cache = self.gru_step_forward(xh[:, t, :], h[:, t - 1, :] if t > 0 else h0,
                                                              self.gruParams[layer]['Wzx'],
                                                              self.gruParams[layer]['Wzh'],
                                                              self.gruParams[layer]['bz'],
                                                              self.gruParams[layer]['Wax'],
                                                              self.gruParams[layer]['War'],
                                                              self.gruParams[layer]['ba'],
                                                               )
                cache.append(tmp_cache)
            xh = h 

            self.gruParams[layer]['h']=h
            self.gruParams[layer]['cache'] = cache

        return h

    def gru_backward(self, dh):
        """
        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0, dWzx, dWzh, dbz ,dWax, dWar, dba= None, None, None, None, None, None, None, None
        N,T,H = dh.shape
        x, _, _, _, _, _, _, _,_,_ = self.gruParams[0]['cache'][0]
        D = x.shape[1]

        dh_prevl = dh
        dweights=[]

        for layer in range(self.layersNum-1,-1,-1):
            #得到前向传播保存的cache数组
            cache = self.gruParams[layer]['cache']

            DH = D if layer == 0 else H
            dx = np.zeros((N, T, DH))
            dWzx = np.zeros((DH,2*H))
            dWzh = np.zeros((H,2*H))
            dbz = np.zeros((2 * H))

            dWax = np.zeros((DH,H))
            dWar = np.zeros((H,H))
            dba = np.zeros((H))

            dprev_h = np.zeros((N,H))

            for t in range(T - 1, -1, -1):
                dx[:, t, :], dprev_h,  dWzx_t, dWzh_t, dbz_t,dWax_t, dWar_t, dba_t = self.gru_step_backward(dh_prevl[:, t, :] + dprev_h,
                                                                                      cache[t])  # 注意此处的叠加
                dWzx += dWzx_t
                dWzh += dWzh_t
                dbz += dbz_t

                dWax += dWax_t
                dWar += dWar_t
                dba += dba_t
            dh_prevl=dx

            dweight = (dWzx, dWzh, dbz, dWax, dWar, dba)
            dweights.append(dweight)

        # 返回x误差和各层参数误差
        return dx, dweights

    def gru_step_forward(self, x, prev_h, Wzx, Wzh, bz,Wax, War, ba):
        """
        Inputs:
        - x: Input data, of shape (N, D)
        - prev_h: Previous hidden state, of shape (N, H)
        - prev_c: previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        - cache: Tuple of values needed for backward pass.
        """
        next_h,  cache = None, None

        H = prev_h.shape[1]
        z_hat = np.matmul(x, Wzx) + np.matmul(prev_h, Wzh) + bz

        r = Tools.sigmoid(z_hat[:, :H])
        z = Tools.sigmoid(z_hat[:, H:2 * H])

        a = np.matmul(x,Wax) + np.matmul(r*prev_h,War) + ba

        next_h = prev_h *(1.-z) + z * np.tanh(a)

        cache = (x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a)

        return next_h, cache

    def gru_step_backward(self, dnext_h, cache):
        """
        Inputs:
        - dnext_h: Gradients of next hidden state, of shape (N, H)
        - dnext_c: Gradients of next cell state, of shape (N, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data, of shape (N, D)
        - dprev_h: Gradient of previous hidden state, of shape (N, H)
        - dprev_c: Gradient of previous cell state, of shape (N, H)
        - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba = None, None, None, None, None, None, None, None
        x, prev_h, Wzx, Wzh, Wax, War, z_hat, r, z, a = cache

        N,D = x.shape
        H = dnext_h.shape[1]

        z_hat_H1=z_hat[:, :H]
        z_hat_H2 = z_hat[:, H:2 * H]
        # delta
        tanha = np.tanh(a)
        dh = dnext_h
        da = dh*z*(1.-tanha*tanha)
        dh_prev_1 = dh * (1.-z)
        dz = dh * (z+tanha)
        dz_hat_2 = dz*(z*(1.-z))

        d13 = np.matmul(da,War.T)
        dr = d13 * prev_h
        dx_1 = np.matmul(da,Wax.T)
        dh_prev_2 = d13*r 
        dz_hat_1 = dh_prev_2 * (r * (1. - r))

        dz_hat = np.hstack((dz_hat_1,dz_hat_2))

        dh_prev_3 = np.matmul(dz_hat,Wzh.T)
        dx_2 = np.matmul(dz_hat,Wzx.T)
        dx_3 = np.matmul(dz_hat,Wzx.T)
        dh_prev_4 =np.matmul(dz_hat, Wzh.T)
        dprev_h = dh_prev_1+dh_prev_2+dh_prev_3 + dh_prev_4
        dx = dx_1 + dx_2 +dx_3

        dWax = np.matmul(x.T,da)
        dWar = np.matmul((r*prev_h).T,da)
        dba = np.sum(da,axis=0)

        dWzx = np.matmul(x.T,dz_hat)
        dWzh = np.matmul(prev_h.T,dz_hat)
        dbz = np.sum(dz_hat,axis=0)

        return dx, dprev_h, dWzx, dWzh, dbz, dWax, dWar, dba