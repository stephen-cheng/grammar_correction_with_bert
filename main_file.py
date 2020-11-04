from gector.gec_model import GecBERTModel
from spellchecker import SpellChecker
from part_of_speech import get_part_of_speech
from part_of_speech import meaning
import os


#all arguments for model
current_dir=os.getcwd()
vocab_path=os.path.join(current_dir,"data/output_vocabulary/")
model_path=os.path.join(current_dir,"model/xlnet_0_gector.th") #can add more model here if needed

#default values for model
# max_len=50
# min_len=3
# iterations=5
# min_error_probability=0.66
# min_probability=0.0
# lowercase_tokens=0
# model_name='xlnet'
# special_tokens_fix=0
# confidence=0.35
# is_ensemble=0
# weigths=None


#sentence predictor
def predict_for_sentence(inp_sent, model):
    predictions = []
    cnt_corrections = 0
    batch = []
    batch.append(inp_sent.split())
    if batch:
        preds, cnt, info_for_change = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    after_correction=("\n".join([" ".join(x) for x in predictions]))
    return after_correction, cnt_corrections, info_for_change


#for loading model
def load_model():
	model = GecBERTModel(vocab_path=vocab_path,
    	                 model_paths=[model_path],
        	             min_error_probability=0.66,
            	         model_name='xlnet',
                         max_len=50,
                         min_len=3,
                         iterations=5,
                         min_probability=0.0,
                         lowercase_tokens=0,
                         special_tokens_fix=0,
                         confidence=0.0, #keep it zero on cpu
                         is_ensemble=0,
                         weigths=None
                	    )
	return model


#function for spell checking
def check_spell(sent):
  spell = SpellChecker()

  misspelled = sent.split()
  corrected=[]

  for word in misspelled:
    corrected.append(spell.correction(word))
  sent=" "
  return sent.join(corrected)


#function used for converting inoformation into respective dict
def convert_to_dict(info_list,corrected):
    dictionary={
        'corrected':corrected,
        'errors':{
                    i:info_list[i] for i in range(len(info_list))
                 }
                }
    return dictionary



#function for adding tokens in respective places
def add_tokens(my_sentence,info_for_change):
    org_sent_word=[]                   #changed here
    org_sent_word=my_sentence.split()  #changed here

    word_info=[]

    for info in info_for_change:
        word=[info[0],info[1]]
        if "APPEND" in word[0]:
            if word[1]-1 > -1 :
                word[1]=org_sent_word[word[1]-1]
                word[0]+=" after "
            else:
                word[1]="<BEG>"
                word[0]+=" after "
                continue
        elif "REPLACE" in word[0]:
            word[1]=org_sent_word[word[1]]
            word[0]+=" in place of "
            word[0]=word[0].replace("REPLACE","PUT")
        elif "TRANSFORM" in word[0]:
            word[1]=org_sent_word[word[1]]
            word[0]+=" on "
        else:
            word[1]=org_sent_word[word[1]]
            word[0]+=" "

        word[0]=word[0].replace("$","").replace("_"," ") #comment this for deferentiating tokens easily

        error_correct_sent=word[0]+word[1]      #adding the two parts of error together
                                                #eg. from ['append ? after','him'] to 'append ? after him'

        #made the symbols more understandable
        error_correct_sent=error_correct_sent.replace("VBD",meaning["VBD"]).replace("VBG",meaning["VBG"]).replace("VBZ",meaning["VBZ"]).replace("VBN",meaning["VBN"]).replace("VB",meaning["VB"])

        word_info.append(error_correct_sent)

    return word_info


#function to be used in this file
def check_grammar(my_sentence):
    corrected_sent,cnt_corrections,info_for_change = predict_for_sentence(my_sentence, model)

    word_info=add_tokens(my_sentence,info_for_change)

    info_dict=convert_to_dict(word_info,corrected_sent)
    return info_dict


#implementing the code here

# model=load_model() #loading model

# while True:
#     my_sentence=input("Enter sentence for grammar check: ")
#     print(check_grammar(my_sentence))

#     # get_part=input("Wanna get part of speech ? (y/n) : ") #for getting part of speech from text
#     # if(get_part=='y'):
#     #     print(get_part_of_speech(my_sentence))
    
#     again=input("Wanna do it again? (y/n) : ")
#     if(again=='n'):
#         break



#for downloading xlnet follow https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gector.th