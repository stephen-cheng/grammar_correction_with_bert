import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def data_load(filename):
	# Load the dataset into a pandas dataframe.
	df = pd.read_csv(filename, delimiter='\t', header=None, 
		names=['sentence_source', 'label', 'label_notes', 'sentence'])

	# Report the number of sentences.
	print('Number of training sentences: {:,}\n'.format(df.shape[0]))

	# Get the lists of sentences and their labels.
	sentences = df.sentence.values
	labels = df.label.values
	return sentences, labels


def bert_tokenize(sentences, labels):
	# Load the BERT tokenizer.
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_ids = []
	attention_masks = []

	# For every sentence...
	for sent in sentences:
	    # `encode_plus` will:
	    #   (1) Tokenize the sentence.
	    #   (2) Prepend the `[CLS]` token to the start.
	    #   (3) Append the `[SEP]` token to the end.
	    #   (4) Map tokens to their IDs.
	    #   (5) Pad or truncate the sentence to `max_length`
	    #   (6) Create attention masks for [PAD] tokens.
	    encoded_dict = tokenizer.encode_plus(
	                        sent,                      # Sentence to encode.
	                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
	                        max_length = 64,           # Pad & truncate all sentences.
	                        pad_to_max_length = True,
	                        return_attention_mask = True,   # Construct attn. masks.
	                        return_tensors = 'pt',     # Return pytorch tensors.
	                   )
	    
	    # Add the encoded sentence to the list.    
	    input_ids.append(encoded_dict['input_ids'])
	    
	    # And its attention mask (simply differentiates padding from non-padding).
	    attention_masks.append(encoded_dict['attention_mask'])

	# Convert the lists into tensors.
	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)

	# Print sentence 0, now as a list of IDs.
	print('Original: ', sentences[0])
	print('Token IDs:', input_ids[0])

	return input_ids, attention_masks, labels


def data_split(input_ids, attention_masks, labels):
	# Combine the training inputs into a TensorDataset.
	dataset = TensorDataset(input_ids, attention_masks, labels)

	# Create a 90/10 train-validation split.

	# Calculate the number of samples to include in each set.
	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size

	# Divide the dataset by randomly selecting samples.
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	print('{:>5,} training samples'.format(train_size))
	print('{:>5,} validation samples'.format(val_size))

	return train_dataset, val_dataset


def data_iter(train_dataset, val_dataset):
	# The DataLoader needs to know our batch size for training, so we specify it 
	# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
	# size of 16 or 32.
	batch_size = 32

	# Create the DataLoaders for our training and validation sets.
	# We'll take training samples in random order. 
	train_dataloader = DataLoader(
	            train_dataset,  # The training samples.
	            sampler = RandomSampler(train_dataset), # Select batches randomly
	            batch_size = batch_size # Trains with this batch size.
	        )

	# For validation the order doesn't matter, so we'll just read them sequentially.
	validation_dataloader = DataLoader(
	            val_dataset, # The validation samples.
	            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
	            batch_size = batch_size # Evaluate with this batch size.
	        )
	return train_dataloader, validation_dataloader


def main(filename):
	sentences, labels = data_load(filename)
	input_ids, attention_masks, labels = bert_tokenize(sentences, labels)
	train_dataset, val_dataset = data_split(input_ids, attention_masks, labels)
	train_dataloader, validation_dataloader = data_iter(train_dataset, val_dataset)
	return train_dataloader, validation_dataloader

if __name__ == '__main__':
	filename = "dataset/cola_public/raw/in_domain_train.tsv"
	train_dataloader, validation_dataloader = main(filename)



