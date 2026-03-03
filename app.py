from flask import Flask, render_template, request, jsonify
from docx import Document
import PyPDF2
import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

DATASET_PATH = "dataset"

# ------------------------
# Text Preprocessing
# ------------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)

# ------------------------
# Load Dataset Files
# ------------------------
def load_dataset():
    documents = []
    filenames = []

    print("Loading dataset from:", DATASET_PATH)

    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            print("Found file:", file)
            if file.endswith(".txt") or file.endswith(".java"):
                file_path = os.path.join(root, file)
                print("Reading:", file_path)

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    documents.append(preprocess_text(content))
                    filenames.append(file)

    print("Total files loaded:", len(documents))
    return documents, filenames

dataset_docs, dataset_names = load_dataset()
print("Total dataset files loaded:", len(dataset_docs))

# ------------------------
# Plagiarism Check
# ------------------------
@app.route("/check", methods=["POST"])
def check_plagiarism():

    if len(dataset_docs) == 0:
        return jsonify({
            "similarity_percentage": 0,
            "matched_file": "Dataset empty!"
        })

    user_text = ""

    # If text input exists
    if "text" in request.form and request.form["text"].strip() != "":
        user_text = request.form["text"]

    # If file uploaded
    if "file" in request.files:
        file = request.files["file"]

        if file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                user_text += page.extract_text()

        elif file.filename.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                user_text += para.text

    user_text = preprocess_text(user_text)

    if user_text.strip() == "":
        return jsonify({
            "similarity_percentage": 0,
            "matched_file": "No content provided!"
        })

    all_docs = dataset_docs + [user_text]

    vectorizer = TfidfVectorizer(
        token_pattern=r'\b\w+\b',
        ngram_range=(1,2)
    )

    tfidf_matrix = vectorizer.fit_transform(all_docs)

    similarity_scores = cosine_similarity(
        tfidf_matrix[-1],
        tfidf_matrix[:-1]
    )

    max_score = similarity_scores.max()
    max_index = similarity_scores.argmax()

    return jsonify({
        "similarity_percentage": round(float(max_score * 100), 2),
        "matched_file": dataset_names[max_index]
    })

# ------------------------
# Home Page
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
