from keras.preprocessing import sequence
import numpy as np
from keras.utils import to_categorical
import re
from keras.callbacks import Callback


def _create_dic(text):
    chars = sorted(list(set(text)))
    char_to_int_dic = dict((c, i) for i, c in enumerate(chars))
    int_to_char_dic = dict((i, c) for i, c in enumerate(chars))
    return char_to_int_dic, int_to_char_dic


def _isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def _remove_stopwords(text, stopwords):
    word_list = text.split()
    filtered_words = [x for x in word_list if x not in stopwords]
    return ' '.join(filtered_words)


def preprocess_data(path, lowercase=False, show=True, additional_clean=False, min_len=2, only_english=True, stopwords=[]):
    data = open(path, 'r').read()
    data = data.lower() if lowercase else data
    data = re.sub(r'[^\w\s]', ' ',data) if additional_clean else data

    examples = data.split("\n")
    examples = [re.sub(' +', ' ', x).strip() for x in examples]
    examples = [x for x in examples if _isEnglish(x)] if only_english else examples
    examples = [_remove_stopwords(x, stopwords) for x in examples] if stopwords else examples
    examples = [x for x in examples if len(x) >= min_len]
    char_to_int_dic, int_to_char_dic = _create_dic("\n".join(examples))

    if show:
        data_size, vocab_size = len("\n".join(examples)), len(sorted(list(set("\n".join(examples)))))
        print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
        print("char_to_ix = ", "\n", int_to_char_dic, "\n")
        print("ix_to_char = ", "\n", char_to_int_dic)
        print(examples[:5])

    return examples, char_to_int_dic, int_to_char_dic


def _pad_data(X, Y, longest_sentence):
    X = sequence.pad_sequences(X, maxlen=longest_sentence, padding="post")
    Y = sequence.pad_sequences(Y, maxlen=longest_sentence, padding="post")
    return X, Y


def create_dataset(examples, char_to_int_dic, method="one-hot"):
    X = []
    Y = []
    for index in range(len(examples)):
        lineX = [char_to_int_dic[ch] for ch in examples[index]]
        X.append(lineX)
        lineY = lineX[1:] + [char_to_int_dic["\n"]]
        Y.append(lineY)

    vocab_size = len(char_to_int_dic)
    longest_sentence = len(max(X, key=len)) + 1

    X, Y = _pad_data(X, Y, longest_sentence)
    if method == "one-hot":
        X_onehot = to_categorical(X, vocab_size)
        Y_onehot = to_categorical(Y, vocab_size)
    return X_onehot, Y_onehot, longest_sentence, vocab_size


class generate_text(Callback):
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