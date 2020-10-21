import os
from os import listdir
from os.path import join
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file) as f:
        book = f.read()
    return book


def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    return text


def save_data(text_list, filename):
	with open(filename, 'w') as f:
		for line in text_list:
			f.write(line+'\n')
	print('Done')


if __name__ == "__main__":
	path = 'data/20books/'
	book_files = [f for f in listdir(path) if f.endswith('.rtf')]
	books = []

	# load data
	for book in book_files:
	    books.append(load_data(path+book))

	# clean data
	clean_books = []
	for book in books:
	    clean_books.append(clean_text(book))

	# split lines
	sentences = []
	for book in clean_books:
	    for sentence in book.split('. '):
	    	sentence.replace(",", "")
	    	sentence.replace('"', '')
	    	sentence.replace(":", "")
	    	sentence.replace("'", "")
	    	sentences.append(sentence)
	print("There are {} sentences.".format(len(sentences)))

	# Limit the data we will use to train our model
	max_length = 32
	min_length = 2
	good_sentences = []
	for sentence in sentences:
	    if len(sentence.split()) <= max_length and len(sentence.split()) >= min_length:
	        good_sentences.append(sentence)
	print("{} sentences are used to our model.".format(len(good_sentences)))

	# data split
	train_val, test = train_test_split(good_sentences, test_size=0.1)
	train, val = train_test_split(train_val, test_size=0.1)

	# save the data
	train_file = 'data/train.txt'
	save_data(train, train_file)
	val_file = 'data/val.txt'
	save_data(val, val_file)
	test_file = 'data/test.txt'
	save_data(test, test_file)


