text = '''The future king is the prince
Daughter is the princess
Son is the prince
Only a man can be a king
Only a woman can be a queen
The princess will be a queen
Queen and king rule the realm
The prince is a strong man
The princess is a beautiful woman
The royal family is the king and queen and their children
Prince is only a boy now
A boy will be a man'''

# text = '''Tokyo likes funny stories
# She likes to hear funny stories
# She likes to tell funny stories
# She told her mom a funny story
# When she finished she waited for her mom to laugh
# Mom why are not you laughing
# That was a funny story Tokyo said
# Oh I am sorry her mom said
# Sometimes you think something is funny but someone else thinks it is not funny
# So Tokyos mom did not laugh at Tokyos story
# Tokyo told the same story to her younger sister
# Her younger sister laughed at the story'''

# text = '''First I wake up
# Then I get dressed
# I walk to school
# I do not ride a bike
# I do not ride the bus
# I like to go to school
# It rains
# I do not like rain
# I eat lunch
# I eat a sandwich and an apple
# I play outside
# I like to play
# I read a book
# I like to read books
# I walk home
# I do not like walking home
# My mother cooks soup for dinner
# The soup is hot
# Then I go to bed
# I do not like to go to bed'''

# text = '''Jack was hungry
# He walked to the kitchen
# He got out some eggs
# He took out some oil
# He placed a skillet on the stove
# Next he turned on the heat
# He poured the oil into the skillet
# He cracked the eggs into a bowl
# He stirred the eggs
# Then he poured them into the hot skillet
# He waited while the eggs cooked
# They cooked for two minutes
# He heard them cooking
# They popped in the oil
# Next Jack put the eggs on a plate
# He placed the plate on the dining room table
# Jack loved looking at his eggs
# They looked pretty on the white plate
# He sat down in the large wooden chair
# He thought about the day ahead
# He ate the eggs with a spoon
# They were good
# He washed the plate with dishwashing soap
# Then he washed the pan
# He got a sponge damp
# Finally he wiped down the table
# Next Jack watched TV'''

# text = '''Maria was learning to add numbers
# She liked to add numbers
# It was easy to add numbers
# She could add one and one
# She knew that one and one are two
# She knew that two and two are four
# She knew that three and three are six
# But that was it
# She did not know what four and four are
# She asked her mom
# Her mom told her that four and four are eight
# Oh now I know Maria said
# I am four years old now
# In four more years I will be eight
# Maria was a fast learner
# She was not a slow learner'''

# text = '''Elliot a brilliant but highly unstable young cybersecurity engineer and vigilante hacker becomes a key figure in a complex game of global dominance when he and his shadowy allies try to take down the corrupt corporation he works for
# Elliot is a brilliant introverted young programmer who works as a cybersecurity engineer by day and vigilante hacker by night
# He also happens to be suffering from a strange condition similar to schizophrenia which he futilely tries to keep under control by regularly taking both legal and illegal drugs and visiting his therapist
# When a strange feisty young woman named Darlene and a secretive middle-aged man calling himself MrRobot who claims to be the mysterious leader of an underground hacking group known as F-Society offer Elliot a chance to take his vigilantism to the next level and help them take down E-Corp the corrupt multi-national financial company that Elliot works for and likes to call Evil Corp Elliot finds himself at the crossroads
# MrRobot who has personal reasons for wanting to take down E-Corp also reveals that he already has one ally an even more mysterious secretive and highly dangerous shadowy hacking group known only as Dark Army
# Meanwhile Elliot 's childhood and only friend Angela who blames E-Corp for the death of their parents tries to take down E-Corp legally by joining their ranks and trying to dig up evidence of their corruption from the inside
# A wild card in this scheme becomes Tyrell Wellick an unhinged psychopathic E-Corp yuppie originally from Scandinavia who has a very unusual relationship with his dominant and ambitious wife Joanna
# After many twists and turns MrRobot 's plan is finally put in motion with catastrophic unintended results
# But that 's just the end of the beginning of the real story
# MrRobot follows Elliot a young programmer who works as a cybersecurity engineer by day and as a vigilante hacker by night
# Elliot finds himself at a crossroads when the mysterious leader of an underground hacker group recruits him to destroy the firm he is paid to protect
# Compelled by his personal beliefs Elliot struggles to resist the chance to take down the multinational CEOs he believes are running and ruining the world
# Eventually he realizes that a global conspiracy does exist but not the one he expected
# Young antisocial computer programmer Elliot works as a cybersecurity engineer during the day but at night he is a vigilante hacker
# He is recruited by the mysterious leader of an underground group of hackers to join their organization
# Elliot 's task is to Help bring down corporate America including the company he is paid to protect which presents him with a moral dilemma
# Although he works for a corporation his personal beliefs make it hard to resist the urge to take down the heads of multinational companies that he believes are running and ruining the world
# In the year 2015 an incredibly unstable cybersecurity engineer named Elliot is recruited by a shady group of hackers to bring down the most powerful corporation in the history of civilization meanwhile a dark project whose purpose could bring about profound implications for society itself nears its completion
# Elliot 's internal struggles surrounding his upbringing and personal life intertwine with his present external challenges as part of fsociety
# Being very introverted Elliot fails to express emotions and determine for himself what is real and what is not a question that is also left somewhat ambiguous to the viewer
# In particular Elliot struggles to remember important facts about his close relatives
# Elliot is ultimately revealed to be partly responsible for the death of neighbour and close friend Shayla though accidental
# His search for other chemicals to mix with his morphine addiction led Shayla to become reliant on a criminal psychopath
# Elliot manages to help imprison the criminal before being forced to help free him on the promise that Shayla would be left unharmed a promise that was not kept
# Meanwhile in Evil Corp Tyrell Wellick is desperate to be promoted to CTO
# The determined young employee also appears to have an unusual relationship with his similarly ambitious wife with his pending promotion their only real common purpose
# Failing to get the promotion out of desperation Tyrell kills the new CTO 's wife showing increasingly psychopathic tendencies
# Eventually with the hard work of fsociety and with help from other hackers in China the large scale hack on Evil Corp is carried out successfully and all debt data held by the giant corporation is lost
# it 's hard to know exactly to what extent Elliot is himself wholly fsociety but the group certainly appears to be ultimately based mainly around Elliot and his sister Darlene'''

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

np.random.seed(0)

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