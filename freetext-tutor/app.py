from flask import Flask, Response, request, render_template, jsonify, make_response
from flask_cors import CORS
import sys, os
import spacy
nlp = spacy.load('de')
app = Flask(__name__)

CORS(app)

def getDependencyTree(text):
    doc = nlp(text)
    return doc.print_tree()[0]


@app.route('/')
def hello():
    return "Hello World!"

@app.route('/dep-tree', methods=['POST', 'GET'])
def image():
    if request.method == 'POST':
        app.logger.info("POST request")
        try:
            text = request.json["text"]
            text += " returned!"
            addr = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
            status_code=200
            return jsonify(
                success=True,
                tree=getDependencyTree(text),
                # addr=addr,
                statusCode=200)
        except Exception as e:
            app.logger.error("ERROR:", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            app.logger.error(exc_type, fname, exc_tb.tb_lineno)
            return jsonify(success=False, statusCode=500)
        return make_response(response, status_code)

if __name__ == '__main__':
    app.run()
