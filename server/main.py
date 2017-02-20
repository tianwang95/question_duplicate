import csv
from flask import Flask, request
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
app = Flask(__name__)

MODEL_FILE = '../model.hdf5'
QUESTION_FILE = '../dataset/raw/quora_duplicate_questions.tsv'
INDEX_DIR = 'index'

# Global state
schema = Schema(question=TEXT(stored=True))
index = create_in("index", schema)
index = open_dir("index")

def fill_index(index):
    writer = index.writer()
    questions = set({})
    with open(QUESTION_FILE, 'r') as f:
        question_tsv = csv.reader(f, delimiter='\t')
        next(question_tsv) # Skip headers
        for row in question_tsv:
            if row[1] not in questions:
                writer.add_document(question=row[3])
            if row[2] not in questions:
                writer.add_document(question=row[4])
            questions.add(row[1])
            questions.add(row[2])
    writer.commit()
    return questions

def answer(question):
    with index.searcher() as searcher:
        parser = QueryParser("question", index.schema)
        query = parser.parse(question)
        results = searcher.search(query)
        return results[0]['question']

@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/query', methods=['POST'])
def api_query():
    if request.headers['Content-Type'] != 'text/plain':
        return "415 Unsupported Media Type"
    question = str(request.data)
    print("Received Question: " + question)
    return answer(question)

if __name__ == '__main__':
    # fill_index(index)
    app.run()
