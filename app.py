from flask import Flask           # import flask
from flask import jsonify
from main_file import load_model
from main_file import predict_for_sentence
from main_file import convert_to_dict
from main_file import add_tokens


app = Flask(__name__)             # create an app instance

#loading model
model=load_model()				  #loading model from main_file

@app.route("/")                   # at the end point /
def hello():                      # call method hello
    return "Hello World!"         # which returns "hello world"



@app.route("/<sentence>")              #this is same as the grammar check function in main file
def correct_sent(sentence):
	corrected_sent,cnt_corrections, info_for_change = predict_for_sentence(sentence, model)

	word_info=add_tokens(sentence,info_for_change)

	info_dict=convert_to_dict(word_info,corrected_sent)
	return jsonify(info_dict)

if __name__ == "__main__":        # on running python app.py
    app.run(debug=True, host='127.0.0.1', port=7000)

