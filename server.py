from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from summarize import summarizeText

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

MAX_LEN = 3000


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get('text')
    if (len(text) < MAX_LEN):
        summary = summarizeText(text)
        return {"summary": summary}
    else:
        abort(413, description="The input text exceeds 3000 characters!")
