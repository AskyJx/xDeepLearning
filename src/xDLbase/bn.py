"""
Created: May 2018
@author: JerryX
Find more : https://www.zhihu.com/people/xu-jerry-82
"""

import numpy as np
import time

import logging.config

# create logger
import os
exec_abs = os.getcwd()
log_conf = exec_abs + '/config/logging.conf'
logging.config.fileConfig(log_conf)
logger = logging.getLogger('bn')


# batch normalization layer
# with optimizer adopter
class BNLayer(object):

    def __init__(self,LName,eps,miniBatchesSize,channel,i_size,activator,optimizerCls,optmParams, dataType):
        self.name = LName
        self.miniBatchesSize = miniBatchesSize
        self.i_size = i_size
        self.channel = channel
        self.activator = activator
        self.eps = eps
        self.optimizerObj = optimizerCls(optmParams, dataType)
        self.dataType = dataType
        self.beta = np.zeros((channel,i_size,i_size),dataType)
        self.gamma = np.ones((channel,i_size,i_size),dataType)
        self.mu_accum = np.zeros((channel,i_size,i_size),dataType)
        self.var_accum = np.ones((channel,i_size,i_size),dataType)
        self.decay_accum = 0.95  # 滑动平均衰减率
        self.cache = ()
        self.out = []
        self.deltaPrev = []  
        self.deltaOri = [] 

    # 滑动平均衰减累计mini-batch的mean和variance
    def mov_avg_accum(self,mu,var):
        decay = self.decay_accum

        self.mu_accum = decay * self.mu_accum + (1-decay) * mu
        self.var_accum = decay * self.var_accum + (1-decay) * var

    def inference(self,x):
        oriOut = self.bnForward_inf(x,self.gamma,self.beta,self.eps)
        self.out = self.activator.activate(oriOut)
        return self.out

    def fp(self,x):
        oriOut,self.cache = self.bnForward_tr(x,self.gamma,self.beta,self.eps)
        self.out = self.activator.activate(oriOut)
        return self.out

    # for training
    def bnForward_tr(self, x, gamma, beta, eps):

      mu = np.mean(x, axis=0)
      xmu = x - mu
      var = np.mean(xmu **2, axis = 0)

      self.mov_avg_accum(mu, var)

      ivar = 1./np.sqrt(var + eps)
      xhat = xmu * ivar
      out = gamma*xhat + beta

      cache = (xhat,gamma,ivar)

      return out, cache

    # 已优化，不使用分步骤方式
    def bnBackward(self,dout, cache):

      st = time.time()
      logger.debug('bnBk start')

      xhat,gamma,ivar = cache

      N = dout.shape[0]

      dbeta = np.sum(dout, axis=0)
      dgamma = np.sum(dout*xhat, axis=0)

      dxhat = dout * gamma
      dx = 1./N* ivar * (N*dxhat-np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat,axis=0))

      logger.debug('bnBk end, %s s', (time.time()-st) )

      return dx, dgamma, dbeta


    # for predict, Unbiased Estimation
    def bnForward_inf(self, x, gamma, beta, eps):

      N = self.miniBatchesSize

      var = N / (N - 1) * self.var_accum
      tx = (x - self.mu_accum) /np.sqrt(var + eps)
      out = gamma * tx + beta

      return out

    def bp(self,input,delta,lrt):
        self.deltaOri = self.activator.bp(delta, self.out)

        self.deltaPrev, dgamma, dbeta = self.bnBackward(self.deltaOri, self.cache)

        weight=(self.gamma,self.beta)
        dweight = (dgamma,dbeta)
        self.optimizerObj.getUpdWeights(weight,dweight, lrt)

        return self.deltaPrev