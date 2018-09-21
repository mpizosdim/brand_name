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


class GenerateNames(Callback):
    def __init__(self, char_to_int, int_to_char, num_names, max_size, vocab_size):
        self.char_to_int_dic = char_to_int
        self.int_to_char_dic = int_to_char
        self.num_names = num_names
        self.max_size = max_size
        self.vocab_size = vocab_size

    def on_epoch_end(self, epoch, logs={}):
        for i in range(0, self.num_names):
            stop = False
            index = np.random.randint(1, self.vocab_size)
            ch = self.int_to_char_dic[index]
            counter = 1
            target_seq = np.zeros((1, self.max_size, self.vocab_size))
            target_seq[0, 0, self.char_to_int_dic[ch]] = 1.
            while stop == False and counter < self.max_size:
                probs = self.model.predict(target_seq, verbose=0)[:, counter-1, :]
                index = np.random.choice(range(self.vocab_size), p=probs.ravel())
                c = self.int_to_char_dic[index]
                if c == '\n':
                    stop = True
                else:
                    ch = ch + c
                    target_seq[0, counter, self.char_to_int_dic[c]] = 1.
                    counter = counter + 1

            print(ch)


class NamerLstm(object):
    def __init__(self):
        self.X = None
        self.Y = None
        self.stopwords = []
        self.char2int_dict = None
        self.int2char_dict = None
        self.longest_sentence = None
        self.vocab_size = None
        self.model = None
        self.hidden_size = None

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords
        pass

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
        examples = [self.__remove_stopwords(x) for x in examples] if stopwords else examples
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

    def build_model(self, hidden_size):
        input_layer = Input(shape=(self.longest_sentence, self.vocab_size,))
        lstm_layer = LSTM(hidden_size, return_sequences=True)(input_layer)
        dense_layer = TimeDistributed(Dense(self.vocab_size))(lstm_layer)
        actication_layer = TimeDistributed(Activation('softmax'))(dense_layer)
        model = Model(inputs=[input_layer], outputs=[actication_layer])
        self.hidden_size = hidden_size
        self.model = model

    def fit(self, lr=0.01, epoch_size=2, names_to_output=10):
        rmsp = RMSprop(lr=lr)
        self.model.compile(optimizer=rmsp, loss='categorical_crossentropy', metrics=["categorical_accuracy"])
        gen = GenerateNames(self.char2int_dict, self.int2char_dict, names_to_output, self.longest_sentence, self.vocab_size)
        checkpoint = ModelCheckpoint("model_%s_%s" % (self.hidden_size, lr), monitor='loss', verbose=1, save_best_only=True, mode='min')
        self.model.fit(x=self.X, y=self.Y, epochs=epoch_size, callbacks=[gen, checkpoint])

    def summary(self):
        print('vocab size is: %d' % self.vocab_size)
        print('size of data is: %s' % self.X.shape)


if __name__ == '__main__':
    hidden_size = 250
    number_of_names = 10
    stopwords = ['shoes', 'paris', 'london', 'milano', 'jeans', 'eyewear', 'jewelry']

    namerAlgo = NamerLstm()
    namerAlgo.set_stopwords(stopwords)
    namerAlgo.create_dataset("brandnames.csv", lowercase=True, additional_clean=True, min_len=2, only_english=True)
    namerAlgo.build_model(hidden_size)
    namerAlgo.fit()