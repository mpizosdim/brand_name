from keras.layers import Input, Dense
from keras.models import Model
from utils import preprocess_data, create_dataset, generate_text
from keras.layers import SimpleRNN, LSTM, TimeDistributed, Activation
from keras.optimizers import RMSprop


def initialize_simple_model(hidden_size, longest_sentence, vocab_size):
    input_layer = Input(shape=(longest_sentence, vocab_size,))
    rnn_layer = SimpleRNN(hidden_size, return_sequences=True)(input_layer)
    dense_layer = Dense(vocab_size, activation='softmax')(rnn_layer)
    model = Model(inputs=[input_layer], outputs=[dense_layer])
    return model


def initialize_lstm_model(hidden_size, longest_sentence, vocab_size):
    input_layer = Input(shape=(longest_sentence, vocab_size,))
    lstm_layer = LSTM(hidden_size, return_sequences=True)(input_layer)
    dense_layer = TimeDistributed(Dense(vocab_size))(lstm_layer)
    actication_layer = TimeDistributed(Activation('softmax'))(dense_layer)
    model = Model(inputs=[input_layer], outputs=[actication_layer])
    return model


if __name__ == '__main__':
    hidden_size = 250
    number_of_names = 10
    stopwords = ['shoes', 'paris', 'london', 'milano', 'jeans', 'eyewear', 'jewelry']
    examples, char_to_int_dic, int_to_char_dic = preprocess_data("brandnames.csv", True, True, True, stopwords=stopwords)
    x, y, longest_sentence, vocab_size = create_dataset(examples, char_to_int_dic)
    gen = generate_text(char_to_int_dic, int_to_char_dic, number_of_names, longest_sentence, vocab_size)
    model = initialize_lstm_model(hidden_size, longest_sentence, vocab_size)
    rmsp = RMSprop(lr=0.01)
    model.compile(optimizer=rmsp, loss='categorical_crossentropy', metrics=["categorical_accuracy"])

    epoch_size = 100
    max_size = 20
    model.fit(x=x, y=y, epochs=epoch_size, callbacks=[gen])