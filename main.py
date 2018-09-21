from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.layers import LSTM, TimeDistributed, Activation
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import re
import pickle
from keras.models import load_model as load_model_keras
import matplotlib.pyplot as plt


class GenerateNames(Callback):
    def __init__(self, lstm_model, num_names):
        self.lstm_model = lstm_model
        self.num_names = num_names

    def on_epoch_end(self, epoch, logs={}):
        for i in range(0, self.num_names):
            stop = False
            index = np.random.randint(1, self.lstm_model.vocab_size)
            ch = self.lstm_model.int2char_dict[index]
            counter = 1
            target_seq = np.zeros((1, self.lstm_model.longest_sentence, self.lstm_model.vocab_size))
            target_seq[0, 0, self.lstm_model.char2int_dict[ch]] = 1.
            while stop == False and counter < self.lstm_model.longest_sentence:
                probs = self.lstm_model.model.predict(target_seq, verbose=0)[:, counter - 1, :]
                #probs = self.model.predict(target_seq, verbose=0)[:, counter - 1, :] NOT SURE WHICH IS CORRECT
                index = np.random.choice(range(self.lstm_model.vocab_size), p=probs.ravel())
                c = self.lstm_model.int2char_dict[index]
                if c == '\n':
                    stop = True
                else:
                    ch = ch + c
                    target_seq[0, counter, self.lstm_model.char2int_dict[c]] = 1.
                    counter = counter + 1

            print(ch)


class NamerLstm(object):
    def __init__(self, hidden_size):
        self.X = None
        self.Y = None
        self.stopwords = []
        self.char2int_dict = None
        self.int2char_dict = None
        self.longest_sentence = None
        self.vocab_size = None
        self.model = None
        self.hidden_size = hidden_size
        self.lstm_layers = None

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords
        pass

    def save_info(self, path):
        additional_info = {"vocab_size": self.vocab_size, "longest_sentence": self.longest_sentence}
        with open(path, 'wb') as f:
            pickle.dump([self.char2int_dict, self.int2char_dict, additional_info], f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, path):
        self.model = load_model_keras(path)

    def load_info(self, path):
        with open(path, 'rb') as f:
            info = pickle.load(f)
            self.char2int_dict = info[0]
            self.int2char_dict = info[1]
            self.vocab_size = info[2]["vocab_size"]
            self.longest_sentence = info[2]["longest_sentence"]

    def __pad_data(self, X, Y, longest_sentence):
        X = sequence.pad_sequences(X, maxlen=longest_sentence, padding="post")
        Y = sequence.pad_sequences(Y, maxlen=longest_sentence, padding="post")
        return X, Y

    def __preprocess_data(self, path, lowercase=False, additional_clean=False, min_len=2, only_english=True):
        data = open(path, 'r').read()
        data = data.lower() if lowercase else data
        data = re.sub(r'[^\w\s]', ' ', data) if additional_clean else data

        examples = data.split("\n")
        examples = [re.sub(' +', ' ', x).strip() for x in examples]
        examples = [x for x in examples if self.__is_english(x)] if only_english else examples
        examples = [self.__remove_stopwords(x) for x in examples] if self.stopwords else examples
        examples = [x for x in examples if len(x) >= min_len]
        return examples

    def create_dataset(self, path, lowercase=False, additional_clean=False, min_len=2, only_english=True):
        examples = self.__preprocess_data(path, lowercase, additional_clean, min_len, only_english)
        char2int_dic, int2char_dic = self.__create_dic("\n".join(examples))
        self.char2int_dict = char2int_dic
        self.int2char_dict = int2char_dic
        X = []
        Y = []
        for index in range(len(examples)):
            lineX = [self.char2int_dict[ch] for ch in examples[index]]
            X.append(lineX)
            lineY = lineX[1:] + [self.char2int_dict["\n"]]
            Y.append(lineY)

        vocab_size = len(self.char2int_dict)
        longest_sentence = len(max(X, key=len)) + 1

        X, Y = self.__pad_data(X, Y, longest_sentence)
        X_onehot = to_categorical(X, vocab_size)
        Y_onehot = to_categorical(Y, vocab_size)
        self.X = X_onehot
        self.Y = Y_onehot
        self.longest_sentence = longest_sentence
        self.vocab_size = vocab_size
        pass

    def __remove_stopwords(self, text):
        word_list = text.split()
        filtered_words = [x for x in word_list if x not in self.stopwords]
        return ' '.join(filtered_words)

    @staticmethod
    def __create_dic(text):
        chars = sorted(list(set(text)))
        char_to_int_dic = dict((c, i) for i, c in enumerate(chars))
        int_to_char_dic = dict((i, c) for i, c in enumerate(chars))
        return char_to_int_dic, int_to_char_dic

    @staticmethod
    def __is_english(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def build_model(self, number_of_lstm_layers=1):
        input_layer = Input(shape=(self.longest_sentence, self.vocab_size,))
        lstm_layer = LSTM(self.hidden_size, return_sequences=True, name='lstm_layer_0')(input_layer)
        for i in range(number_of_lstm_layers-1):
                name = 'lstm_layer_{0}'.format(i+1)
                lstm_layer = LSTM(self.hidden_size, return_sequences=True, name=name)(lstm_layer)
        dense_layer = TimeDistributed(Dense(self.vocab_size))(lstm_layer)
        actication_layer = TimeDistributed(Activation('softmax'))(dense_layer)
        model = Model(inputs=[input_layer], outputs=[actication_layer])
        self.lstm_layers = number_of_lstm_layers
        self.model = model

    @staticmethod
    def __plot_loss(history):
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

    def fit(self, lr=0.01, epoch_size=100, names_to_output_per_epoch=10, model_path=""):
        rmsp = RMSprop(lr=lr)
        self.model.compile(optimizer=rmsp, loss='categorical_crossentropy', metrics=["categorical_accuracy"])
        gen = GenerateNames(self, names_to_output_per_epoch)
        checkpoint = ModelCheckpoint(model_path + "model_%s_%s_%s" % (self.hidden_size, lr, self.lstm_layers), monitor='loss', verbose=1, save_best_only=True, mode='min')
        history = self.model.fit(x=self.X, y=self.Y, epochs=epoch_size, callbacks=[gen, checkpoint])
        self.__plot_loss(history)

    def get_names(self, num_of_names):
        for i in range(0, num_of_names):
            stop = False
            index = np.random.randint(1, self.vocab_size)
            ch = self.int2char_dict[index]
            counter = 1
            target_seq = np.zeros((1, self.longest_sentence, self.vocab_size))
            target_seq[0, 0, self.char2int_dict[ch]] = 1.
            while stop == False and counter < self.longest_sentence:
                probs = self.model.predict(target_seq, verbose=0)[:, counter - 1, :]
                index = np.random.choice(range(self.vocab_size), p=probs.ravel())
                c = self.int2char_dict[index]
                if c == '\n':
                    stop = True
                else:
                    ch = ch + c
                    target_seq[0, counter, self.char2int_dict[c]] = 1.
                    counter = counter + 1

            print(ch)

    def summary(self):
        print('vocab size is: %d' % self.vocab_size)
        print('size of data is: %s' % self.X.shape)


# ====== mains ==========


def main_train():
    hidden_size = 250
    epoch_size = 2
    lstm_layers = 2
    stopwords = ['shoes', 'paris', 'london', 'milano', 'jeans', 'eyewear', 'jewelry', 'inc']

    namerAlgo = NamerLstm(hidden_size)
    namerAlgo.set_stopwords(stopwords)
    namerAlgo.create_dataset("brandnames.csv", lowercase=True, additional_clean=True, min_len=2, only_english=True)
    namerAlgo.build_model(lstm_layers)
    namerAlgo.fit(epoch_size=epoch_size)
    namerAlgo.save_info("info.pkl")


def main_load():
    hidden_size = 250
    lstm_layers = 2
    namerAlgo = NamerLstm(hidden_size)
    namerAlgo.load_info("info.pkl")
    namerAlgo.build_model(lstm_layers)
    namerAlgo.load_model("model_250_0.01_2")
    namerAlgo.get_names(10)

if __name__ == '__main__':
    main_load()
