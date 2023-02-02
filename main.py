from flask import Flask, request, render_template
from cerberus import Validator

from model import predictOnInput

app = Flask(__name__)


@app.route("/")
@app.route("/form")
def form():
    return render_template("form.html")


@app.route("/submit", methods=["POST"])
def submit():
    input_validator = Validator({
        'text': {
            'type': 'string',
            'required': True,
            'empty': False
        },
        'limit': {
            'type': 'number',
            'required': True,
            'empty': False
        }
    })

    neg, pos, neut = predictOnInput(
        request.form.get('text'),
        False,
        True,
        int(request.form.get('limit'))
    )

    return render_template("result.html", neglen=len(neg), negatives=neg, poslen=len(pos), postives=pos,
                           neutlen=len(neut), neutrals=neut)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
