import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Embedding, Dense

def main():
    # load dataset
    lines = pd.read_csv("input/Hindi_English_Truncated_Corpus.csv", encoding='utf-8')
    lines.drop_duplicates(inplace=True)
    lines = lines.sample(n=25000, random_state=42)
    print("dataset shape:", lines.shape)

    # fill missing values
    lines['english_sentence'] = lines['english_sentence'].fillna("").astype(str)
    lines['hindi_sentence'] = lines['hindi_sentence'].fillna("").astype(str)

    # clean text: lowercase, remove punctuation, numbers, extra spaces
    def clean_text(text):
        text = text.lower()
        text = re.sub("'", '', text)
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        text = re.sub(" +", " ", text)
        return text

    lines['english_sentence'] = lines['english_sentence'].apply(clean_text)
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(clean_text)

    # add start and end tokens to Hindi sentences
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: 'START_ ' + x + ' _END')

    # create vocabulary
    all_eng_words = set()
    all_hin_words = set()
    for eng in lines['english_sentence']:
        for w in eng.split():
            all_eng_words.add(w)
    for hin in lines['hindi_sentence']:
        for w in hin.split():
            all_hin_words.add(w)

    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_hin_words))

    num_encoder_tokens = len(input_words) + 1
    num_decoder_tokens = len(target_words) + 1

    input_token_index = {word: i + 1 for i, word in enumerate(input_words)}
    target_token_index = {word: i + 1 for i, word in enumerate(target_words)}
    reverse_target_index = {i: word for word, i in target_token_index.items()}

    # get maximum sequence lengths
    max_len_src = max([len(txt.split()) for txt in lines['english_sentence']])
    max_len_tar = max([len(txt.split()) for txt in lines['hindi_sentence']])
    print("max english length:", max_len_src)
    print("max hindi length:", max_len_tar)
    print("vocab sizes:", num_encoder_tokens, num_decoder_tokens)

    # prepare sequences as integers
    def encode_sequences(texts, token_index, max_len):
        data = np.zeros((len(texts), max_len), dtype='int32')
        for i, sentence in enumerate(texts):
            for t, word in enumerate(sentence.split()):
                if word in token_index:
                    data[i, t] = token_index[word]
        return data

    encoder_input_data = encode_sequences(lines['english_sentence'], input_token_index, max_len_src)
    decoder_input_data = encode_sequences(lines['hindi_sentence'], target_token_index, max_len_tar)

    X_train, X_test, y_train, y_test = train_test_split(
        encoder_input_data, decoder_input_data, test_size=0.2, random_state=42
    )

    # build gru seq2seq model
    latent_dim = 256

    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(num_encoder_tokens, latent_dim, mask_zero=True, name='embedding_enc')(encoder_inputs)
    encoder_gru = GRU(latent_dim, return_state=True, name='gru_enc')
    encoder_outputs, encoder_state = encoder_gru(enc_emb)

    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb = Embedding(num_decoder_tokens, latent_dim, mask_zero=True, name='embedding_dec')(decoder_inputs)
    decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True, name='gru_dec')
    decoder_outputs, _ = decoder_gru(dec_emb, initial_state=encoder_state)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='dense_output')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    model.summary()

    # shift decoder input by 1 to create target
    decoder_target_data = np.zeros_like(decoder_input_data)
    decoder_target_data[:, 0:-1] = decoder_input_data[:, 1:]

    # train model
    model.fit(
        [X_train, y_train],
        np.expand_dims(decoder_target_data, -1),
        batch_size=64,
        epochs=20,
        validation_split=0.2
    )

    # inference models for prediction
    encoder_model = Model(encoder_inputs, encoder_state)

    decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
    decoder_inputs_inf = Input(shape=(None,), name='decoder_inputs_inf')

    dec_emb_layer = model.get_layer('embedding_dec')
    decoder_gru_layer = model.get_layer('gru_dec')
    decoder_dense_layer = model.get_layer('dense_output')

    dec_emb_inf = dec_emb_layer(decoder_inputs_inf)
    dec_outputs_inf, state_h_inf = decoder_gru_layer(dec_emb_inf, initial_state=decoder_state_input_h)
    dec_outputs_inf = decoder_dense_layer(dec_outputs_inf)

    decoder_model = Model(
        [decoder_inputs_inf, decoder_state_input_h],
        [dec_outputs_inf, state_h_inf]
    )

    # decode function
    def decode_sequence(input_seq):
        state_value = encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target_token_index['START_']

        decoded_sentence = ''
        stop_condition = False

        while not stop_condition:
            output_tokens, h = decoder_model.predict([target_seq, state_value])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = reverse_target_index.get(sampled_token_index, '')

            decoded_sentence += ' ' + sampled_word

            if sampled_word == '_END' or len(decoded_sentence.split()) > 30:
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            state_value = h

        return decoded_sentence.replace('_END', '').strip()

    # test on one random sentence
    i = np.random.randint(0, len(X_test))
    input_seq = X_test[i:i+1]
    decoded = decode_sequence(input_seq)
    eng_sentence = lines['english_sentence'].iloc[i]

    print("\ninput english sentence:", eng_sentence)
    print("predicted hindi translation:", decoded)

if __name__ == "__main__":
    main()
