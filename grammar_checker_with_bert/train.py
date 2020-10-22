import torch
import time
import os
import datetime
import random
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
from data_prepare import main
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score


def bert_model():
	# Load BertForSequenceClassification, the pretrained BERT model with a single 
	# linear classification layer on top. 
	model = BertForSequenceClassification.from_pretrained(
	    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
	    num_labels = 2, # The number of output labels--2 for binary classification.
	                    # You can increase this for multi-class tasks.   
	    output_attentions = False, # Whether the model returns attentions weights.
	    output_hidden_states = False, # Whether the model returns all hidden-states.
	)

	if torch.cuda.is_available():
		device = torch.device("cuda")
		# Tell pytorch to run this model on the GPU.
		model.cuda()
	else:
		device = torch.device("cpu")


	# Get all of the model's parameters as a list of tuples.
	params = list(model.named_parameters())

	print('The BERT model has {:} different named parameters.\n'.format(len(params)))

	print('==== Embedding Layer ====\n')

	for p in params[0:5]:
	    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

	print('\n==== First Transformer ====\n')

	for p in params[5:21]:
	    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

	print('\n==== Output Layer ====\n')

	for p in params[-4:]:
	    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

	return model, device


def model_params(model, train_dataloader):
	# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
	# I believe the 'W' stands for 'Weight Decay fix"
	optimizer = AdamW(model.parameters(),
	                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
	                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
	                )
	# Number of training epochs. The BERT authors recommend between 2 and 4. 
	# We chose to run for 4, but we'll see later that this may be over-fitting the
	# training data.
	epochs = 4

	# Total number of training steps is [number of batches] x [number of epochs]. 
	# (Note that this is not the same as the number of training samples).
	total_steps = len(train_dataloader) * epochs

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(optimizer, 
	                                            num_warmup_steps = 0, # Default value in run_glue.py
	                                            num_training_steps = total_steps)
	return optimizer, epochs, scheduler


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# This training code is based on the `run_glue.py` of github.com/huggingface/transformers
def model_train(epochs, train_dataloader, validation_dataloader, model, device, optimizer, scheduler):
	# Set the seed value all over the place to make this reproducible.
	seed_val = 42

	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)

	# We'll store a number of quantities such as training and validation loss, 
	# validation accuracy, and timings.
	training_stats = []

	# Measure the total training time for the whole run.
	total_t0 = time.time()

	# For each epoch...
	for epoch_i in range(0, epochs):

		# ========================================
	    #               Training
	    # ========================================
	    
	    # Perform one full pass over the training set.

	    print("")
	    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
	    print('Training...')

	    # Measure how long the training epoch takes.
	    t0 = time.time()

	    # Reset the total loss for this epoch.
	    total_train_loss = 0

	    # Put the model into training mode. Don't be mislead--the call to 
	    # `train` just changes the *mode*, it doesn't *perform* the training.
	    # `dropout` and `batchnorm` layers behave differently during training
	    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
	    model.train()

	    # For each batch of training data...
	    for step, batch in enumerate(train_dataloader):

	        # Progress update every 40 batches.
	        if step % 40 == 0 and not step == 0:
	            # Calculate elapsed time in minutes.
	            elapsed = format_time(time.time() - t0)
	            
	            # Report progress.
	            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

	        # Unpack this training batch from our dataloader. 
	        #
	        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
	        # `to` method.
	        #
	        # `batch` contains three pytorch tensors:
	        #   [0]: input ids 
	        #   [1]: attention masks
	        #   [2]: labels 
	        b_input_ids = batch[0].to(device)
	        b_input_mask = batch[1].to(device)
	        b_labels = batch[2].to(device)

	        # Always clear any previously calculated gradients before performing a
	        # backward pass. PyTorch doesn't do this automatically because 
	        # accumulating the gradients is "convenient while training RNNs". 
	        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
	        model.zero_grad()        

	        # Perform a forward pass (evaluate the model on this training batch).
	        # The documentation for this `model` function is here: 
	        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
	        # It returns different numbers of parameters depending on what arguments
	        # arge given and what flags are set. For our useage here, it returns
	        # the loss (because we provided labels) and the "logits"--the model
	        # outputs prior to activation.
	        loss, logits = model(b_input_ids, 
	                             token_type_ids=None, 
	                             attention_mask=b_input_mask, 
	                             labels=b_labels)

	        # Accumulate the training loss over all of the batches so that we can
	        # calculate the average loss at the end. `loss` is a Tensor containing a
	        # single value; the `.item()` function just returns the Python value 
	        # from the tensor.
	        total_train_loss += loss.item()

	        # Perform a backward pass to calculate the gradients.
	        loss.backward()

	        # Clip the norm of the gradients to 1.0.
	        # This is to help prevent the "exploding gradients" problem.
	        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

	        # Update parameters and take a step using the computed gradient.
	        # The optimizer dictates the "update rule"--how the parameters are
	        # modified based on their gradients, the learning rate, etc.
	        optimizer.step()

	        # Update the learning rate.
	        scheduler.step()

	    # Calculate the average loss over all of the batches.
	    avg_train_loss = total_train_loss / len(train_dataloader)            
	    
	    # Measure how long this epoch took.
	    training_time = format_time(time.time() - t0)

	    print("")
	    print("  Average training loss: {0:.2f}".format(avg_train_loss))
	    print("  Training epcoh took: {:}".format(training_time))
	        
	    # ========================================
	    #               Validation
	    # ========================================
	    # After the completion of each training epoch, measure our performance on
	    # our validation set.

	    print("")
	    print("Running Validation...")

	    t0 = time.time()

	    # Put the model in evaluation mode--the dropout layers behave differently
	    # during evaluation.
	    model.eval()

	    # Tracking variables 
	    total_eval_accuracy = 0
	    total_eval_loss = 0
	    nb_eval_steps = 0

	    # Evaluate data for one epoch
	    for batch in validation_dataloader:
	        
	        # Unpack this training batch from our dataloader. 
	        #
	        # As we unpack the batch, we'll also copy each tensor to the GPU using 
	        # the `to` method.
	        #
	        # `batch` contains three pytorch tensors:
	        #   [0]: input ids 
	        #   [1]: attention masks
	        #   [2]: labels 
	        b_input_ids = batch[0].to(device)
	        b_input_mask = batch[1].to(device)
	        b_labels = batch[2].to(device)
	        
	        # Tell pytorch not to bother with constructing the compute graph during
	        # the forward pass, since this is only needed for backprop (training).
	        with torch.no_grad():

	            # Forward pass, calculate logit predictions.
	            # token_type_ids is the same as the "segment ids", which 
	            # differentiates sentence 1 and 2 in 2-sentence tasks.
	            # The documentation for this `model` function is here: 
	            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
	            # Get the "logits" output by the model. The "logits" are the output
	            # values prior to applying an activation function like the softmax.
	            (loss, logits) = model(b_input_ids, 
	                                   token_type_ids=None, 
	                                   attention_mask=b_input_mask,
	                                   labels=b_labels)
	            
	        # Accumulate the validation loss.
	        total_eval_loss += loss.item()

	        # Move logits and labels to CPU
	        logits = logits.detach().cpu().numpy()
	        label_ids = b_labels.to('cpu').numpy()

	        # Calculate the accuracy for this batch of test sentences, and
	        # accumulate it over all batches.
	        total_eval_accuracy += flat_accuracy(logits, label_ids)
	        

	    # Report the final accuracy for this validation run.
	    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
	    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

	    # Calculate the average loss over all of the batches.
	    avg_val_loss = total_eval_loss / len(validation_dataloader)
	    
	    # Measure how long the validation run took.
	    validation_time = format_time(time.time() - t0)
	    
	    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
	    print("  Validation took: {:}".format(validation_time))

	    # Record all statistics from this epoch.
	    training_stats.append(
	        {
	            'epoch': epoch_i + 1,
	            'Training Loss': avg_train_loss,
	            'Valid. Loss': avg_val_loss,
	            'Valid. Accur.': avg_val_accuracy,
	            'Training Time': training_time,
	            'Validation Time': validation_time
	        }
	    )

	print("")
	print("Training complete!")
	print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

	return training_stats, model


def train_summary(training_stats):
	# Display floats with two decimal places.
	pd.set_option('precision', 2)

	# Create a DataFrame from our training statistics.
	df_stats = pd.DataFrame(data=training_stats)

	# Use the 'epoch' as the row index.
	df_stats = df_stats.set_index('epoch')

	# A hack to force the column headers to wrap.
	#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
	return df_stats


def train_visual(df_stats):
	# Use plot styling from seaborn.
	sns.set(style='darkgrid')

	# Increase the plot size and font size.
	sns.set(font_scale=1.5)
	plt.rcParams["figure.figsize"] = (12,6)

	# Plot the learning curve.
	plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
	plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

	# Label the plot.
	plt.title("Training & Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.xticks([1, 2, 3, 4])
	plt.savefig("result/model_loss.png", dpi=150)
	# plt.show()


def model_eval(filename, model):
	# Load the dataset into a pandas dataframe.
	df = pd.read_csv(filename, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

	# Report the number of sentences.
	print('Number of test sentences: {:,}\n'.format(df.shape[0]))

	# Create sentence and label lists
	sentences = df.sentence.values
	labels = df.label.values

	# Tokenize all of the sentences and map the tokens to thier word IDs.
	input_ids = []
	attention_masks = []

	# For every sentence...
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
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

	# Set the batch size.  
	batch_size = 32  

	# Create the DataLoader.
	prediction_data = TensorDataset(input_ids, attention_masks, labels)
	prediction_sampler = SequentialSampler(prediction_data)
	prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

	# Prediction on test set

	print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

	# Put model in evaluation mode
	model.eval()

	# Tracking variables 
	predictions , true_labels = [], []

	# Predict 
	for batch in prediction_dataloader:
	  # Add batch to GPU
	  batch = tuple(t.to(device) for t in batch)
	  
	  # Unpack the inputs from our dataloader
	  b_input_ids, b_input_mask, b_labels = batch
	  
	  # Telling the model not to compute or store gradients, saving memory and 
	  # speeding up prediction
	  with torch.no_grad():
	      # Forward pass, calculate logit predictions
	      outputs = model(b_input_ids, token_type_ids=None, 
	                      attention_mask=b_input_mask)

	  logits = outputs[0]

	  # Move logits and labels to CPU
	  logits = logits.detach().cpu().numpy()
	  label_ids = b_labels.to('cpu').numpy()
	  
	  # Store predictions and true labels
	  predictions.append(logits)
	  true_labels.append(label_ids)

	print('    DONE.')
	print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))
	return predictions, true_labels


def mcc_eval(predictions, true_labels):
	matthews_set = []
	# Evaluate each test batch using Matthew's correlation coefficient
	print('Calculating Matthews Corr. Coef. for each batch...')

	# For each input batch...
	for i in range(len(true_labels)):
	  
	  # The predictions for this batch are a 2-column ndarray (one column for "0" 
	  # and one column for "1"). Pick the label with the highest value and turn this
	  # in to a list of 0s and 1s.
	  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
	  
	  # Calculate and store the coef for this batch.  
	  matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
	  matthews_set.append(matthews)

	# Create a barplot showing the MCC score for each batch of test samples.
	ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)
	plt.title('MCC Score per Batch')
	plt.ylabel('MCC Score (-1 to +1)')
	plt.xlabel('Batch #')

	plt.savefig("result/mcc_eval.png", dpi=150)
	# plt.show()

	# Combine the results across all batches. 
	flat_predictions = np.concatenate(predictions, axis=0)
	# For each sample, pick the label (0 or 1) with the higher score.
	flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
	# Combine the correct labels for each batch into a single list.
	flat_true_labels = np.concatenate(true_labels, axis=0)
	# Calculate the MCC
	mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
	print('Total MCC: %.3f' % mcc)


def model_save(model):
	# Saving best-practices: if you use defaults names for the model using from_pretrained()
	output_dir = 'model/'

	# Create output directory if needed
	if not os.path.exists(output_dir):
	    os.makedirs(output_dir)

	print("Saving model to %s" % output_dir)

	# Save a trained model, configuration and tokenizer using `save_pretrained()`.
	# They can then be reloaded using `from_pretrained()`
	# Take care of distributed/parallel training
	model_to_save = model.module if hasattr(model, 'module') else model  
	model_to_save.save_pretrained(output_dir)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	tokenizer.save_pretrained(output_dir)

	# Good practice: save your training arguments together with the trained model
	# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
	print('Model is saved!')


if __name__ == '__main__':
	filename = "dataset/cola_public/raw/in_domain_train.tsv"
	train_dataloader, validation_dataloader = main(filename)
	model, device = bert_model()
	optimizer, epochs, scheduler = model_params(model, train_dataloader)
	training_stats, model = model_train(epochs, train_dataloader, validation_dataloader, model, device, optimizer, scheduler)
	df_stats = train_summary(training_stats)
	train_visual(df_stats)
	filename = "dataset/cola_public/raw/out_of_domain_dev.tsv"
	predictions, true_labels = model_eval(filename, model)
	mcc_eval(predictions, true_labels)
	model_save(model)





