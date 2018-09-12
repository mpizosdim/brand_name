from keras.layers import Input, Dense
from keras.models import Model
from utils import preprocess_data, create_dataset, generate_name
from keras.layers import SimpleRNN
from keras.optimizers import SGD


def initialize_simple_model(hidden_size, longest_sentence, vocab_size):
    input_layer = Input(shape=(longest_sentence, vocab_size,))
    rnn_layer = SimpleRNN(hidden_size, return_sequences=True)(input_layer)
    dense_layer = Dense(vocab_size, activation='softmax')(rnn_layer)
    model = Model(inputs=[input_layer], outputs=[dense_layer])
    return model

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

