import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim

import random


class LSTMnn(nn.Module):
    def __init__(self, num_categories, hidden_size, num_layers):
        super(LSTMnn, self).__init__()
        self.lstm = nn.LSTM(num_categories, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_categories)

    def forward(self, batch, hidden_state):
        outputs, hidden_state = self.lstm(batch, hidden_state)
        outputs = self.fc(outputs)
        return outputs, hidden_state


# Maps each music note in 'filename' to an integer
def createMap(filename):
    char2int = dict()
    int2char = dict()
    i = 1
    with open(filename) as f:
        while True:
            c = f.read(1)
            if not c:
                break
            if c not in char2int:
                char2int[c] = i
                int2char[i] = c
                i += 1
    return char2int, int2char


# One-hot encodes song using previously created mapping from notes to integers
def encodeSong(songString, char2int):
    encoding = torch.zeros((len(songString), len(char2int) + 1))
    for i in range(len(songString)):
        if songString[i] in char2int:
            ind = char2int[songString[i]]
        else:
            ind = 0
        encoding[i][ind] = 1
    return encoding


# One-hot encodes all songs in 'filename' and returns list of encoded songs
def encodeFile(filename, char2int):
    songs = []
    curSong = ""
    with open(filename) as f:
        for line in f:
            curSong += line
            if line == "<end>\n":
                songs.append(encodeSong(curSong, char2int))
                curSong = ""
    return songs


# Converts one-hot encoded song back into music notes using mapping from integers to notes
def decodeSong(rawEncoding, int2char):
    encoding = rawEncoding.reshape((-1, len(int2char) + 1))
    song = ""
    for e in encoding:
        if np.argmax(e) in int2char:
            song += int2char[np.argmax(e)]
    return song


# Create a list of one-hot encoding of songs
char2int, int2char = createMap("train.txt")
songs = encodeFile("train.txt", char2int)

# Input includes each valid note value plus an extra mapping for any unknown characters
input_size = len(char2int) + 1
hidden_size = 100
num_layers = 1

# Training variables
epochs = 100
notes_per_batch = 100

# Creates network, backprop system with learning rate of 0.001, and cross entropy loss criterion
lstm = LSTMnn(input_size, hidden_size, num_layers)
optimizer = optim.Adam(lstm.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


print(songs[0])
# Training epochs
for e in range(epochs):

    # Shuffles training data
    indices = list(range(len(songs)))
    random.shuffle(indices)

    for i in indices:

        hidden_state = (
            torch.zeros((1, 1, hidden_size)),
            torch.zeros((1, 1, hidden_size)),
        )
        song = songs[i]
        for j in range(math.ceil(len(song) / notes_per_batch)):
            optimizer.zero_grad()
            batch = song[
                j * notes_per_batch : min((j + 1) * notes_per_batch, len(song) - 1)
            ]
            batch = batch.reshape((-1, input_size))
            teacher = song[
                j * notes_per_batch + 1 : min((j + 1) * notes_per_batch + 1, len(song))
            ].type(torch.LongTensor)
            teacher = torch.reshape((-1, input_size))
            print(teacher.size())
            outputs, hidden_state = lstm.forward(batch, hidden_state)
            print(outputs)
            print(outputs.size())
            print(teacher.size())
            loss = criterion(outputs, teacher)
            loss.backward()
            optimizer.step()
            print(j)
