from flask import Flask, request, jsonify, abort, send_file
from flask_cors import CORS
from summarize import summarizeText
from spellcheck import check_spelling
from dictionary import word_meaning
from io import BytesIO
from fpdf import FPDF

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

MAX_LEN = 4000


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get('text')

    if (len(text) < MAX_LEN):
        summary = summarizeText(text)
        return {"summary": summary}
    else:
        abort(413, description="The input text exceeds 3000 characters!")


@app.route("/spellcheck", methods=["POST"])
def spellcheck():
    data = request.get_json()
    text = data.get('text')

    if (len(text) < MAX_LEN):
        misspelled = check_spelling(text)
        return {"misspelled": misspelled}
    else:
        abort(413, description="The input text exceeds 3000 characters!")


@app.route("/dictionary", methods=["POST"])
def get_meaning():
    data = request.get_json()
    text = data.get("text")

    return {"meaning": word_meaning(text)}


@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    # Get the text from the request body
    text = request.json['text']
    summary = request.json['summary']

    # Create a new PDF object
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=14, style='BU')

    # Add Input header
    pdf.cell(0, 10, txt="Input:", ln=1)

    # Set font and size for the input text
    pdf.set_font("Arial", size=10)

    # Split the input text into separate lines
    lines = text.splitlines()

    # Output each line as a block of text using the MultiCell method
    for line in lines:
        pdf.multi_cell(0, 6, line, align="J")

    pdf.set_font("Arial", size=14, style='BU')

    # Add Output header
    pdf.cell(0, 10, txt="Output:", ln=1)

    # Set font and size for the summary text
    pdf.set_font("Arial", size=10)

    lines = summary.splitlines()
    for line in lines:
        pdf.multi_cell(0, 6, line, align="J")

    # Get the PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')

    # Send the PDF as a file to the client
    return send_file(
        BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name="output.pdf"
    )
