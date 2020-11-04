import nltk 
# nltk.download('stopwords')  #if doesnt work download all these first
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 

meaning_with_example = {
	"CC" : "coordinating conjunction",
	"CD" : "cardinal digit",
	"DT" : "determiner",
	"EX" : "existential there (like: “there is” … think of it like “there exists”)",
	"FW" : "foreign word",
	"IN" : "preposition/subordinating conjunction",
	"JJ" : "adjective ‘big’",
	"JJR": "adjective, comparative ‘bigger’",
	"JJS": "adjective, superlative ‘biggest’",
	"LS" : "list marker 1)",
	"MD" : "modal could, will",
	"NN" : "noun, singular ‘desk’",
	"NNS": "noun plural ‘desks’",
	"NNP": "proper noun, singular ‘Harrison’",
	"NNPS": "proper noun, plural ‘Americans’",
	"PDT": "predeterminer ‘all the kids’",
	"POS": "possessive ending parent‘s",
	"PRP": "personal pronoun I, he, she",
	"PRP$": "possessive pronoun my, his, hers",
	"RB" : "adverb very, silently,",
	"RBR": "adverb, comparative better",
	"RBS": "adverb, superlative best",
	"RP" : "particle give up",
	"TO" : "to go ‘to‘ the store.",
	"UH" : "interjection errrrrrrrm",
	"VB" : "verb, base form take",
	"VBD": "verb, past tense took",
	"VBG": "verb, gerund/present participle taking",
	"VBN": "verb, past participle taken",
	"VBP": "verb, sing. present, non-3d take",
	"VBZ": "verb, 3rd person sing. present takes",
	"WDT": "wh-determiner which",
	"WP" : "wh-pronoun who, what",
	"WP$": "possessive wh-pronoun whose",
	"WRB": "wh-abverb where, when",
	"," : "comma",
	"." : "full stop"
}


meaning = {
	"CC" : "coordinating conjunction",
	"CD" : "cardinal digit",
	"DT" : "determiner",
	"EX" : "existential there",
	"FW" : "foreign word",
	"IN" : "preposition/subordinating conjunction",
	"JJ" : "adjective",
	"JJR": "adjective, comparative",
	"JJS": "adjective, superlative",
	"LS" : "list marker",
	"MD" : "modal could, will",
	"NN" : "noun singular",
	"NNS": "noun plural",
	"NNP": "proper noun, singular",
	"NNPS": "proper noun, plural",
	"PDT": "predeterminer",
	"POS": "possessive ending",
	"PRP": "personal pronoun",
	"PRP$": "possessive pronoun",
	"RB" : "adverb ",
	"RBR": "adverb, comparative ",
	"RBS": "adverb, superlative ",
	"RP" : "particle ",
	"TO" : "to go ‘to‘ the store.",
	"UH" : "interjection",
	"VB" : "verb base form ",
	"VBD": "verb past tense ",
	"VBG": "verb gerund/present participle",
	"VBN": "verb past participle ",
	"VBP": "verb sing. present",
	"VBZ": "verb 3rd person sing. present ",
	"WDT": "wh-determiner which",
	"WP" : "wh-pronoun who, what",
	"WP$": "possessive wh-pronoun whose",
	"WRB": "wh-abverb where, when"
}


def get_part_of_speech(sentence):
	cleaned=[]
	tokenized = sent_tokenize(sentence) 
	for i in tokenized: 
	    wordsList = nltk.word_tokenize(i) 
	    wordsList = [w for w in wordsList if not w in stop_words] 
	    tagged = nltk.pos_tag(wordsList)
	    for pair in tagged:
	    	c_pair=[]
	    	c_pair.append(pair[0])
	    	try :
	    		c_pair.append(meaning[pair[1]])
	    	except :
	    		c_pair.append("Punctuation")
	    	cleaned.append(c_pair)
	return cleaned

#print(get_part_of_speech("Sukanya, Rajib and Naba are my good friends."))