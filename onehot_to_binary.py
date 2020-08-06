import numpy as np
import random
import time
import struct

def read():
    data = []
    num = np.arange(10, dtype=np.uint8)
    for i in num:
        x = np.zeros(10)
        x[i] = 1
        y = np.unpackbits(i, count=4, bitorder='little').astype(float)
        data.append([x.reshape(1,10), y.reshape(1,4)])
    return data

class layer:
    def __init__(self, inp_size, out_size):
        self.ETA = 0.001
        self.b = np.random.rand(1, out_size)
        self.w = np.random.rand(inp_size, out_size)
        self.bgrad = np.zeros_like(self.b)
        self.wgrad = np.zeros(self.w.shape)
    def forward(self, inp):
        self.x = inp
        self.y = np.dot(self.x, self.w) + self.b
        return self.y
    def compgrad(self, ygrad):
        self.bgrad += ygrad
        self.wgrad += np.dot(self.x.T, ygrad)
        return np.dot(ygrad, self.w.T)
    def backprop(self):
        self.w -= self.ETA * self.wgrad
        self.b -= self.ETA * self.bgrad
        self.bgrad = 0
        self.wgrad = 0

def train():
    EPOCHS = 1000
    BATCHSIZE = 1
    traindata = read()
    datasize = len(traindata)
    for itr in range(EPOCHS):
        random.shuffle(traindata)
        minibatch = (traindata[i : i + BATCHSIZE] for i in range(0, datasize, BATCHSIZE))
        loss = 0.0
        for BATCH in minibatch:
            for x, y in BATCH:
                y_pred = network.forward(x)
                loss += (y_pred - y) * (y_pred - y) / datasize
                y_pred_grad = 2 * (y_pred - y) / BATCHSIZE
                network.compgrad(y_pred_grad)
            network.backprop()
        print('Loss:', loss, 'Epoch:', itr)

def test():
    testdata = read()
    for x, y in testdata:
        n = network.forward(x)
        print(f'x: {x} y: {y} n: {np.around(n)}')

network = layer(10, 4)
train()
test()