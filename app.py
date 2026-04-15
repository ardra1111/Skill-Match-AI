from flask import Flask, request, render_template, jsonify
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text


def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)


def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""


@app.route("/")
def matchresume():
    return render_template('matchresume.html')


@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = clean_text(request.form['job_description'])
        resume_files = request.files.getlist('resumes')

        resumes = []
        filenames = []

        for resume_file in resume_files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filepath)
            text = clean_text(extract_text(filepath))
            if text.strip():
                resumes.append(text)
                filenames.append(resume_file.filename)

        if not resumes or not job_description:
            return jsonify({'error': 'Please upload resumes and enter a job description.'})

        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )

        vectors = vectorizer.fit_transform([job_description] + resumes).toarray()

        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        top_n = min(5, len(similarities))
        top_indices = similarities.argsort()[-top_n:][::-1]
        top_resumes = [filenames[i] for i in top_indices]

        # Raw cosine scores first (0.0 - 1.0 range)
        similarity_scores = [float(similarities[i]) for i in top_indices]

        # Normalize so best = 100%, others relative to it
        max_score = max(similarity_scores)
        normalized_scores = [
            round((score / max_score) * 100, 2) if max_score > 0 else 0
            for score in similarity_scores
        ]

        best_resume = top_resumes[0]

        return jsonify({
            'best_resume': best_resume,
            'top_resumes': top_resumes,
            'similarity_scores': normalized_scores
        })

    return jsonify({'error': 'Invalid request.'})


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
