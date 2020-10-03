from toy_data import texts

text = texts[0] # change index to use another text (range(0, 6))
text = text.lower()
stop_words = ['a', 'and', 'is', 'be', 'will', 'to', 'on', 'but', 'their', 'of', "'s", 'an', 'in', 'are', 'as', 'at', 'by', 'for', 'the']
text = ' '.join([word for word in text.split(' ') if word not in stop_words])
clean_text = [line.split(' ') for line in text.split('\n')]
# print(clean_text)

window = 5
word_lists = []
for text in clean_text:
    for i, word in enumerate(text):
        for w in range(window):
            # Getting the context that is ahead by *window* words
            if i + 1 + w < len(text): 
                word_lists.append([word] + [text[(i + 1 + w)]])
            # Getting the context that is behind by *window* words    
            if i - w - 1 >= 0:
                word_lists.append([word] + [text[(i - w - 1)]])
# print(word_lists)

unique_words = sorted(list(set([word for line in clean_text for word in line])))
n_unique_words = len(unique_words)
# print(unique_words)

import numpy as np
import random

np.random.seed(49)

onehot_vecs = {word: vec for word, vec in zip(unique_words, np.eye(n_unique_words))}
# print(onehot_vecs)

traindata = [[onehot_vecs[x].reshape(1, -1), onehot_vecs[y].reshape(1, -1)] for x, y in word_lists]
# print(train_data)

class Softmax:
    def forward(self, inp):
        self.y = np.exp(inp)
        self.y /= np.sum(self.y)
        return self.y
    def backprop(self, y_grad):
        jacobian_m =  np.diagflat(self.y) - np.dot(self.y.T, self.y)
        return np.dot(y_grad, jacobian_m)

class Relu:
    def forward(self, inp):
        self.x = inp
        return np.maximum(0, inp)
    def backprop(self, y_grad):
        return 1.0 * (self.x > 0) * y_grad

class Leaky_relu:
    def forward(self, inp):
        self.x = inp
        self.leaky_slope = 0.1
        return np.maximum(self.leaky_slope * inp, inp)
    def backprop(self, y_grad):
        return (1.0 * (self.x > 0) + self.leaky_slope * (self.x <= 0)) * y_grad

class Tanh:
    def forward(self, inp):
        self.y = np.tanh(inp)       # self.y = (np.exp(inp) - np.exp(-inp)) / (np.exp(inp) + np.exp(-inp))
        return self.y
    def backprop(self, y_grad):
        return (1 - np.square(self.y)) * y_grad

class Sigmoid:
    def forward(self, inp):
        self.y = 1.0 / (1.0 + np.exp(-inp))
        return self.y
    def backprop(self, y_grad):
        return (self.y * (1 - self.y)) * y_grad

class Layer:
    def __init__(self, inp_size, out_size, ETA, activation=None):
        self.ETA = ETA
        self.b = np.random.randn(1, out_size)
        self.w = np.random.randn(inp_size, out_size)
        self.bgrad = 0.0
        self.wgrad = 0.0
        self.activation = activation() if activation else None
    def forward(self, inp):
        self.x = inp
        self.y = np.dot(self.x, self.w) + self.b
        if self.activation:
            self.y = self.activation.forward(self.y)
        return self.y
    def backprop(self, y_grad):
        if self.activation:
            y_grad = self.activation.backprop(y_grad)
        self.bgrad += y_grad
        self.wgrad += np.dot(self.x.T, y_grad)
        return np.dot(y_grad, self.w.T)
    def step(self):
        self.w -= self.ETA * self.wgrad
        self.b -= self.ETA * self.bgrad
        self.bgrad = 0
        self.wgrad = 0

class Hierachicallayer:
    def __init__(self, inp_size, out_size, ETA, activation=None):
        self.ETA = ETA
        self.blayer = Layer(inp_size, out_size, ETA, activation=Tanh)
        self.wlayers = [Layer(inp_size, inp_size, ETA, activation=Tanh) for i in range(out_size)]
        self.activation = activation() if activation else None
    def forward(self, inp):
        self.x = inp
        self.w = np.concatenate([wlayer.forward(self.x) for wlayer in self.wlayers]).T
        self.b = self.blayer.forward(self.x)
        self.y = np.dot(self.x, self.w) + self.b
        if self.activation:
            self.y = self.activation.forward(self.y)
        return self.y
    def backprop(self, y_grad):
        if self.activation:
            y_grad = self.activation.backprop(y_grad)
        self.bgrad = y_grad
        self.wgrad = np.dot(self.x.T, y_grad)
        self.wgrad = np.split(self.wgrad.T, len(self.wlayers))
        self.xgrad = 0
        for wg, wlayer in zip(self.wgrad, self.wlayers):
            self.xgrad += wlayer.backprop(wg)
        self.xgrad += self.blayer.backprop(self.bgrad)
        self.xgrad += np.dot(y_grad, self.w.T)
        return self.xgrad
    def step(self):
        for wlayer in self.wlayers:
            wlayer.step()
        self.blayer.step()

class Model:
    def __init__(self, net_size=[], ETA=0.03, activation=None):
        self.layers = []
        for i in range(len(net_size) - 1):
            self.layers.append(Layer(net_size[i], net_size[i + 1], ETA, activation))
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    def backprop(self, a_grad):
        for l in reversed(self.layers):
            a_grad = l.backprop(a_grad)
    def step(self):
        for l in reversed(self.layers):
            l.step()
    def append(self, layer):
        self.layers.append(layer)

class BCEloss: # Binary Cross-Entropy Loss
    def forward(self, y, yp):
        return -((y * np.log(yp)) + ((1 - y) * np.log(1 - yp)))
    def backprop(self, y, yp):
        return (-y / yp) + ((1 - y) / (1 - yp))

class CCEloss: # Categorical Cross-Entropy Loss
    def forward(self, y, yp):
        return -np.sum(y * np.log(yp))
    def backprop(self, y, yp):
        return -y / yp

class MSEloss: # Mean Squared Error Loss
    def forward(self, y, yp):
        return (yp - y) ** 2
    def backprop(self, y, yp):
        return 2 * (yp - y)

def train(traindata, BATCH_SIZE=10):
    data_size = len(traindata)
    random.shuffle(traindata)
    batches = (traindata[i : i + BATCH_SIZE] for i in range(0, data_size, BATCH_SIZE))
    loss = 0.0
    loss_fn = BCEloss()
    for batch in batches:
        for x, y in batch:
            yp = network.forward(x)
            loss += loss_fn.forward(y, yp)
            y_grad = loss_fn.backprop(y, yp) / BATCH_SIZE
            network.backprop(y_grad)
        network.step()
    return np.sum(loss) / data_size

def test(testdata):
    correct = 0
    # for x,y in testdata:
    #     n = network.forward(x)
    #     if np.array_equal(np.around(n), y):
    #         correct += 1
    # return correct

EPOCHS = 60
BATCH_SIZE = 20
ETA = 0.3

network = Model([n_unique_words, 2], ETA)
network.append(Layer(2, n_unique_words, ETA, activation=Softmax))

for itr in range(EPOCHS):
    print(f'Loss: {train(traindata, BATCH_SIZE)} Epoch: {itr} ( {test(traindata)} / {len(traindata)} )')

from matplotlib import pyplot as plt
for word, embedding in zip(unique_words, network.layers[0].w):
    plt.scatter(embedding[0], embedding[1])
    plt.annotate(word, (embedding[0], embedding[1]))
plt.show()