from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
from spelling_correction_with_seq2seq_learning.utils import CharacterTable, transform
from spelling_correction_with_seq2seq_learning.utils import restore_model, decode_sequences
from spelling_correction_with_seq2seq_learning.utils import read_text, tokenize



def grammar_checker_model(output_dir, sent):
	# Load the BERT tokenizer.
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained(output_dir)
	model_loaded = BertForSequenceClassification.from_pretrained(output_dir)
	# Let's check it for a given sentence

	encoded_dict = tokenizer.encode_plus(
	                        sent,                      # Sentence to encode.
	                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
	                        max_length = 64,           # Pad & truncate all sentences.
	                        pad_to_max_length = True,
	                        return_attention_mask = True,   # Construct attn. masks.
	                        return_tensors = 'pt',     # Return pytorch tensors.
	                   )
	    
	    # Add the encoded sentence to the list.    
	input_id = encoded_dict['input_ids']
	    
	    # And its attention mask (simply differentiates padding from non-padding).
	attention_mask = encoded_dict['attention_mask']
	input_id = torch.LongTensor(input_id)
	attention_mask = torch.LongTensor(attention_mask)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_loaded = model_loaded.to(device)
	input_id = input_id.to(device)
	attention_mask = attention_mask.to(device)

	with torch.no_grad():
		# Forward pass, calculate logit predictions
		outputs = model_loaded(input_id, token_type_ids=None, attention_mask=attention_mask)

	logits = outputs[0]
	index = logits.argmax()
	if index == 1:
	  print("Gramatically correct")
	  return 0
	else:
	  print("Gramatically in-correct")
	  return 1


def spell_correction_model(model_path, data_path, corpus, test_sentence):
	error_rate = 0.6
	reverse = True
	hidden_size = 512
	sample_mode = 'argmax'
	test_sentence = test_sentence.lower()
	text = read_text(data_path, corpus)
	vocab = tokenize(text)
	vocab = list(filter(None, set(vocab)))
	# `maxlen` is the length of the longest word in the vocabulary
	# plus two SOS and EOS characters.
	maxlen = max([len(token) for token in vocab]) + 2
	train_encoder, train_decoder, train_target = transform(
	    vocab, maxlen, error_rate=error_rate, shuffle=False, train=False)

	tokens = tokenize(test_sentence)
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

	return input_tokens, decoded_tokens


if __name__ == '__main__':

	output_dir = 'grammar_checker_with_bert/model/'
	test_sent = "you doin good work"
	flag = grammar_checker_model(output_dir, test_sent)

	if flag:
		model_path = 'spelling_correction_with_seq2seq_learning/models/seq2seq.h5'
		data_path = 'spelling_correction_with_seq2seq_learning/data'
		corpus = ['train.txt', 'character.txt']
		input_tokens, decoded_tokens = spell_correction_model(model_path, data_path, corpus, test_sent)
	else:

		input_tokens = decoded_tokens = test_sent
	print('---------------------****---------------------')
	print('Input sentence:  ', ' '.join([token for token in input_tokens]))
	print('Corrected sentence:', ' '.join([token for token in decoded_tokens]))




