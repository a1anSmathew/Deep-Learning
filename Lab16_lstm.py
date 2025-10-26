import numpy as np
import pandas as pd
import re, string
from string import digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

def main():
    # load data
    lines = pd.read_csv("input/Hindi_English_Truncated_Corpus.csv", encoding='utf-8')

    # drop duplicates and missing values
    lines.drop_duplicates(inplace=True)
    lines.dropna(subset=['english_sentence', 'hindi_sentence'], inplace=True)
    lines = lines[lines['english_sentence'].apply(lambda x: isinstance(x, str)) &
                  lines['hindi_sentence'].apply(lambda x: isinstance(x, str))]

    # sample 25000 sentences
    lines = lines.sample(n=25000, random_state=42)
    print("dataset shape:", lines.shape)

    # lowercase
    lines['english_sentence'] = lines['english_sentence'].apply(lambda x: x.lower())
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: x.lower())

    # remove apostrophes
    lines['english_sentence'] = lines['english_sentence'].apply(lambda x: re.sub("'", '', x))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("'", '', x))

    # remove punctuation
    spcl_characters = string.punctuation
    lines['english_sentence'] = lines['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in spcl_characters))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in spcl_characters))

    # remove digits
    remove_digits = str.maketrans('', '', digits)
    lines['english_sentence'] = lines['english_sentence'].apply(lambda x: x.translate(remove_digits))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: x.translate(remove_digits))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

    # remove extra spaces
    lines['english_sentence'] = lines['english_sentence'].apply(lambda x: re.sub(" +", " ", x.strip()))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x.strip()))

    # add start and end tokens to Hindi
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: 'START_ ' + x + ' _END')

    # create vocabulary
    all_eng_words = set()
    for eng in lines['english_sentence']:
        all_eng_words.update(eng.split())

    all_hindi_words = set()
    for hin in lines['hindi_sentence']:
        all_hindi_words.update(hin.split())

    # sentence lengths
    lines['length_eng_sentence'] = lines['english_sentence'].apply(lambda x: len(x.split(" ")))
    lines['length_hin_sentence'] = lines['hindi_sentence'].apply(lambda x: len(x.split(" ")))

    max_length_src = max(lines['length_eng_sentence'])
    max_length_tar = max(lines['length_hin_sentence'])

    # word to index mapping
    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_hindi_words))
    num_encoder_tokens = len(input_words)
    num_decoder_tokens = len(target_words) + 1  # add 1 for padding

    input_token_index = dict([(word, i + 1) for i, word in enumerate(input_words)])
    target_token_index = dict([(word, i + 1) for i, word in enumerate(target_words)])

    reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
    reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

    # shuffle dataset
    lines = shuffle(lines)

    # split dataset
    X, y = lines['english_sentence'], lines['hindi_sentence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_samples = len(X_train)
    val_samples = len(X_test)
    batch_size = 128
    epochs = 10

    # batch generator
    def generate_batch(X=X_train, y=y_train, batch_size=128):
        while True:
            for j in range(0, len(X), batch_size):
                encoder_input_data = np.zeros((batch_size, max_length_src), dtype='float32')
                decoder_input_data = np.zeros((batch_size, max_length_tar), dtype='float32')
                decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens), dtype='float32')

                for i, (input_text, target_text) in enumerate(zip(X[j:j + batch_size], y[j:j + batch_size])):
                    for t, word in enumerate(input_text.split()):
                        encoder_input_data[i, t] = input_token_index.get(word, 0)
                    for t, word in enumerate(target_text.split()):
                        if t < len(target_text.split()) - 1:
                            decoder_input_data[i, t] = target_token_index.get(word, 0)
                        if t > 0:
                            decoder_target_data[i, t - 1, target_token_index.get(word, 0)] = 1.

                yield ((encoder_input_data, decoder_input_data), decoder_target_data)

    # define LSTM model
    latent_dim = 300

    # encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(num_encoder_tokens + 1, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # prepare datasets using generator
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generate_batch(X_train, y_train, batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, max_length_src), dtype=tf.float32),
                tf.TensorSpec(shape=(None, max_length_tar), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None, max_length_tar, num_decoder_tokens), dtype=tf.float32)
        )
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: generate_batch(X_test, y_test, batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, max_length_src), dtype=tf.float32),
                tf.TensorSpec(shape=(None, max_length_tar), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None, max_length_tar, num_decoder_tokens), dtype=tf.float32)
        )
    )

    # train model
    model.fit(
        train_dataset,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=val_samples // batch_size,
        verbose=1
    )

    # define inference models
    # encoder model
    encoder_model = Model(encoder_inputs, encoder_states)

    # decoder model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2
    )

    # decoding function
    def decode_sequence(input_seq):
        states_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target_token_index['START_']
        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = reverse_target_char_index.get(sampled_token_index, '')
            decoded_sentence += ' ' + sampled_word

            if sampled_word == '_END' or len(decoded_sentence.split()) > max_length_tar:
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

        return decoded_sentence.replace('_END', '').strip()

    # test one sample
    train_gen = generate_batch(X_train, y_train, batch_size=1)
    k = 0
    (input_seq, actual_output), _ = next(train_gen)
    decoded_sentence = decode_sequence(input_seq)

    print('input english sentence:', X_train.iloc[k])
    print('actual hindi translation:', y_train.iloc[k][6:-4])
    print('predicted hindi translation:', decoded_sentence)

if __name__ == "__main__":
    main()


