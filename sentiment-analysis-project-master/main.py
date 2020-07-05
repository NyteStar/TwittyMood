from flask import Flask, request, jsonify, abort, render_template
from cerberus import Validator

from model.model import process

app = Flask(__name__)

@app.route("/form")
def form():
		return render_template("form.html")

@app.route("/submit", methods = [ "POST" ])
def submit():
		input_validator = Validator({
			'text': {
				'type': 'string',
				'required': True,
				'empty': False
			}
		})

		if not input_validator.validate(request.form):
				abort(400, 'text Must be Correctly Passed.')

		result = process(
			request.form.get('text'),
			False,
			True
		)

		return render_template("result.html", len = len(result), result = result) 
	
if __name__ == "__main__":
		app.run(debug = True)