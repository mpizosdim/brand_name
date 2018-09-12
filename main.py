from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.layers.core import Reshape
from utils import preprocess_data, create_dataset, generate_name
from keras.layers import SimpleRNN
from keras.optimizers import SGD
import numpy as np
import sys
import csv


def initialize_model(n_labels, seq_length=1, mem_units=128, dropout_prob=0.1):
    input_layer = Input(shape=(seq_length, 1, ), dtype='float32', name='input')
    lstm_layer = LSTM(mem_units)(input_layer)
    dropout_layer = Dropout(dropout_prob)(lstm_layer)
    Respahed_layer = Reshape((mem_units, 1,))(dropout_layer)
    lstm_layer2 = LSTM(mem_units)(Respahed_layer)
    dropout_layer2 = Dropout(dropout_prob)(lstm_layer2)
    dense_layer = Dense(n_labels, activation='softmax')(dropout_layer2)
    model = Model(inputscs=[input_layer], outputs=[dense_layer])
    return model


def initialize_simple_model(hidden_size, longest_sentence, vocab_size):
    input_layer = Input(shape=(longest_sentence, vocab_size,))
    rnn_layer = SimpleRNN(hidden_size, return_sequences=True)(input_layer)
    dense_layer = Dense(vocab_size, activation='softmax')(rnn_layer)
    model = Model(inputs=[input_layer], outputs=[dense_layer])
    return model

def initialize_lstm_model():
    pass

if __name__ == '__main__':
    hidden_size = 150
    stopwords = ['shoes', 'paris', 'london', 'milano', 'jeans', 'eyewear', 'jewelry']
    examples, char_to_int_dic, int_to_char_dic = preprocess_data("brandnames.csv", True, True, True, stopwords=stopwords)
    x, y, longest_sentence, vocab_size = create_dataset(examples, char_to_int_dic)
    model = initialize_simple_model(hidden_size, longest_sentence, vocab_size)
    sgd = SGD(lr=0.01, clipvalue=10)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["categorical_accuracy"])

    iters = 30
    epoch_size = 5
    max_size = 20

    for iteration in range(iters):
        print("Iteration %d :" % (iteration + 1))

        model.fit(x=x, y=y, epochs=epoch_size)

        weights = []
        for layer in model.layers:
            w = layer.get_weights()
            weights.append(w)

        Wax, Waa, ba = weights[1]
        Wya, by = weights[2]

        for i in range(7):
            dino_name = generate_name(Waa, Wax, ba, Wya, by, hidden_size, int_to_char_dic, char_to_int_dic, vocab_size, max_size)
            print(''.join(dino_name), end="")

