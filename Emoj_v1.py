import numpy as np
import matplotlib.pyplot as plt
import emoji
import pandas as pd
from keras.utils.np_utils import to_categorical

df_train = pd.read_csv('data/train_emoji.csv', header=None)
df_test = pd.read_csv('data/tesss.csv', header=None)
X_train = df_train[0]
Y_train = df_train[1]
X_test = df_test[0]
Y_test = df_test[1]

# get the maxLen of X_train
maxLen = len(max(X_train, key=len).split())

# one hot
Y_oh_train = to_categorical(Y_train)
Y_oh_test = to_categorical(Y_test)

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('date/glove.6B.50d.txt')

# calculate average
def sentence_to_avg(sentence, word_to_vec_map):
    words = sentence.lower().split()
    avg = np.zeros((len(words), 1))
    total = 0
    
    for w in words:
        total += word_to_vec_map[w]
    avg = total / len(words)
    return avg

# softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# predict
def predict(X, Y, W, b, word_to_vec_map):
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):                       # Loop over training examples
        words = X[j].lower().split()
        
        avg = np.zeros((50,))
        for w in words:
            avg += word_to_vec_map[w]
        avg = avg/len(words)

        # Forward propagation
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
    return pred

# model
def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    np.random.seed(1)
    m = Y.shape[0]                          # number of training examples
    n_y = 5                                 # number of classes  
    n_h = 50                                # dimensions of the GloVe vectors 
    # Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    # Optimization loop
    for t in range(num_iterations):
        for i in range(m):          # Loop over the training examples
            avg = sentence_to_avg(X[i], word_to_vec_map)
            # the softmax layer
            z = np.dot(W, avg) + b
            a = softmax(z)
            # Compute cost
            cost = -np.sum(Y[i] * np.log(a))
            # Compute gradients 
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz
            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py

    return pred, W, b


pred, W, b = model(X_train, Y_train, word_to_vec_map)
print(pred)