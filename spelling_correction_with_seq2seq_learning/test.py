import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd

from utils import CharacterTable, transform
from utils import restore_model, decode_sequences
from utils import read_text, tokenize

error_rate = 0.6
reverse = True
model_path = './models/seq2seq.h5'
hidden_size = 512
sample_mode = 'argmax'
data_path = './data'
corpus = ['train.txt', 'character.txt']

test_sentence = 'We visitd a  ship to do the hsipment businNss.'
test_sentence = test_sentence.lower()


if __name__ == '__main__':
    text = read_text(data_path, corpus)
    vocab = tokenize(text)
    vocab = list(filter(None, set(vocab)))
    # `maxlen` is the length of the longest word in the vocabulary
    # plus two SOS and EOS characters.
    maxlen = max([len(token) for token in vocab]) + 2
    train_encoder, train_decoder, train_target = transform(
        vocab, maxlen, error_rate=error_rate, shuffle=False, train=False)
    df = pd.read_csv("data/test.csv")
    words = list(df.Incorrect)
    cor_words = []
    for w in words:
        word = w.lower()
        tokens = tokenize(word)
        tokens = list(filter(None, tokens))
        nb_tokens = len(tokens)
        misspelled_tokens, _, target_tokens = transform(
            tokens, maxlen, error_rate=error_rate, shuffle=False, train=False)

        input_chars = set(' '.join(train_encoder))
        target_chars = set(' '.join(train_decoder))
        input_ctable = CharacterTable(input_chars)
        target_ctable = CharacterTable(target_chars)

        encoder_model, decoder_model = restore_model(model_path, hidden_size)

        input_tokens, target_tokens, decoded_tokens = decode_sequences(
            misspelled_tokens, target_tokens, input_ctable, target_ctable,
            maxlen, reverse, encoder_model, decoder_model, nb_tokens,
            sample_mode=sample_mode, random=False)

        print('-')
        print('Input sentence:  ', ' '.join([token for token in input_tokens]))
        print('-')
        print('Decoded sentence:', ' '.join([token for token in decoded_tokens]))
        # print('-')
        # print('Target sentence: ', ' '.join([token for token in target_tokens]))

        cor_words.append(' '.join([t for t in decoded_tokens]))

    df['Spellchecker'] = cor_words
    df.to_csv("data/test_result.csv")
    print(df.head())

    # prediction accuracy
    cnt = 0
    for i in range(len(df)):
        if df['Corrected'][i] == df['Spellchecker'][i]:
            cnt += 1
    print("Accuracy without customized words: " + str(cnt / len(df) * 1.0))


