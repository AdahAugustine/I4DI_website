from flask import Flask, render_template, request, jsonify
# import time
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
import time
from chat import get_response


load_dotenv(find_dotenv())
app = Flask(__name__)
CORS(app)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_response(input)
    # time.sleep(10)
    # return "Yes,It works"


if __name__ == '__main__':
    app.run()