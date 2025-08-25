import os
import numpy as np
import sqlite3
import json
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

UPLOAD_FOLDER = 'static/uploads'
DATABASE = 'skin_analysis.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)


def init_db():                                 ##db initialization
    conn =sqlite3.connect(DATABASE)    
    c=conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            disease_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            notes TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS disease_info (
            name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            treatments TEXT NOT NULL,
            prevention TEXT NOT NULL
        )
    ''')

    c.execute("SELECT COUNT(*) FROM disease_info")
    count = c.fetchone()[0]

    if count == 0:
        disease_info = [
            ("Actinic keratosis",
             "A rough, scaly patch on the skin caused by years of sun exposure.",
             "Rough, dry, scaly patches; May be red, tan, pink, or flesh-colored; Usually less than 1 inch in diameter.",
             "Cryotherapy, Topical medications, Photodynamic therapy, Curettage and electrosurgery, Chemical peeling.",
             "Use sunscreen daily; Wear protective clothing; Avoid peak sun hours; Regular skin checks."),
            ("Atopic Dermatitis",
             "A chronic skin condition characterized by itchy, inflamed skin.",
             "Red to brownish-gray patches; Itching, which may be severe; Small, raised bumps; Dry, cracked, scaly skin.",
             "Moisturize regularly; Topical corticosteroids; Immunomodulators; Antihistamines; Light therapy.",
             "Moisturize daily; Identify and avoid triggers; Use mild soaps; Manage stress."),
            ("Benign keratosis",
             "A non-cancerous growth on the skin that develops from skin cells.",
             "Waxy, stuck-on appearance; Light brown to black color; Round or oval shape; Flat or slightly raised.",
             "Often no treatment needed; Cryotherapy; Curettage; Laser therapy.",
             "No specific prevention; Regular skin examinations."),
            ("Dermatofibroma",
             "A common benign skin tumor that presents as a firm nodule.",
             "Firm, round bump; Pink, red, or brown color; May be tender to touch; Usually less than 1 cm in diameter.",
             "Often no treatment needed; Surgical excision if bothersome; Cryotherapy; Steroid injections.",
             "No specific prevention; Protect skin from trauma."),
            ("Melanocytic nevus",
             "A common mole that forms when melanocytes grow in clusters.",
             "Brown or black color; Round shape with well-defined borders; Usually less than 6 mm in diameter; Uniform appearance.",
             "Usually no treatment needed; Surgical removal if suspicious.",
             "Monitor for changes; Protect from sun exposure; Regular skin self-exams."),
            ("Melanoma",
             "The most serious type of skin cancer that develops from pigment-producing cells.",
             "Asymmetrical shape; Irregular border; Varied color; Diameter larger than 6 mm; Evolving size, shape, or color.",
             "Surgical excision; Sentinel lymph node biopsy; Immunotherapy; Targeted therapy; Radiation therapy.",
             "Avoid excessive sun exposure; Use sunscreen; Avoid tanning beds; Regular skin self-exams; Professional skin checks."),
            ("Squamous cell carcinoma",
             "A common form of skin cancer that develops from squamous cells.",
             "Firm, red nodule; Flat sore with crusted surface; New sore or raised area on old scar; Rough, scaly patch on lip.",
             "Surgical excision; Mohs surgery; Radiation therapy; Curettage and electrodesiccation; Topical medications.",
             "Use sunscreen daily; Wear protective clothing; Avoid tanning beds; Check skin regularly."),
            ("Tinea Ringworm Candidiasis",
             "Fungal infections affecting the skin, causing ring-shaped rashes.",
             "Ring-shaped rash; Red, scaly, or cracked skin; Itching; Abnormal nail appearance for nail infections.",
             "Antifungal creams; Oral antifungal medications; Keep affected areas clean and dry.",
             "Practice good hygiene; Don't share personal items; Keep skin dry; Wear clean clothes."),
            ("Vascular lesion",
             "Abnormalities of blood vessels that are visible on the skin.",
             "Red or purple discoloration; May be flat or raised; Can appear anywhere on the body; Sometimes painful.",
             "Laser therapy; Sclerotherapy; Surgical removal; Compression therapy.",
             "Protect skin from sun damage; Avoid trauma to skin; Maintain healthy weight and blood pressure.")
        ]

        c.executemany(
            "INSERT INTO disease_info (name, description, symptoms, treatments, prevention) VALUES (?, ?, ?, ?, ?)",
            disease_info
        )

    conn.commit()
    conn.close()

# Load the model safely
model = load_model("skin_disease_model.keras", compile=False, safe_mode=False)




class_names = ["Actinic keratosis", "Atopic Dermatitis", "Benign keratosis", "Dermatofibroma",
               "Melanocytic nevus", "Melanoma", "Squamous cell carcinoma",
               "Tinea Ringworm Candidiasis", "Vascular lesion"]

# Function to preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to save analysis history
def save_to_history(image_path, disease_class, confidence):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute(
        "INSERT INTO analysis_history (id, image_path, disease_class, confidence, timestamp, notes) VALUES (?, ?, ?, ?, ?, ?)",
        (analysis_id, image_path, disease_class, confidence, timestamp, "")
    )
    conn.commit()
    conn.close()
    return analysis_id

# Function to get analysis history
def get_history(limit=10):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM analysis_history ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    history = [dict(row) for row in c.fetchall()]
    conn.close()
    return history

# Function to get disease information
def get_disease_info(disease_name):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM disease_info WHERE name = ?",
        (disease_name,)
    )
    info = c.fetchone()
    conn.close()
    if info:
        return dict(info)
    return None

# Main route for the application
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(image_path)

        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions) * 100)

        analysis_id = save_to_history(image_path, predicted_class, confidence)
        disease_info = get_disease_info(predicted_class)
        history = get_history(5)

        return render_template("index.html",
                                  image_path=image_path,
                                  predicted_class=predicted_class,
                                  confidence=confidence,
                                  disease_info=disease_info,
                                  history=history,
                                  analysis_id=analysis_id)

    history = get_history(5)
    return render_template("index.html", history=history)

# Route for history page
@app.route("/history")
def history():
    all_history = get_history(limit=100)
    return render_template("history.html", history=all_history)

# API endpoint for history
@app.route("/api/history", methods=["GET"])
def api_history():
    limit = request.args.get('limit', 10, type=int)
    history = get_history(limit)
    return jsonify(history)

# API endpoint for disease info
@app.route("/api/disease/<disease_name>")
def api_disease_info(disease_name):
    info = get_disease_info(disease_name)
    if info:
        return jsonify(info)
    return jsonify({"error": "Disease not found"}), 404

# API endpoint to delete analysis
@app.route("/api/history/<analysis_id>", methods=["DELETE"])
def delete_analysis(analysis_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute("SELECT image_path FROM analysis_history WHERE id = ?", (analysis_id,))
    result = c.fetchone()

    if result:
        image_path = result[0]
        c.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
        conn.commit()

        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({"success": True})

    conn.close()
    return jsonify({"error": "Analysis not found"}), 404

# API endpoint to clear history
@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute("SELECT image_path FROM analysis_history")
    image_paths = [row[0] for row in c.fetchall()]

    c.execute("DELETE FROM analysis_history")
    conn.commit()
    conn.close()

    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    return jsonify({"success": True})

# API endpoint to export analysis as JSON
@app.route("/api/export/<analysis_id>")
def export_analysis(analysis_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    analysis = c.fetchone()

    if not analysis:
        conn.close()
        return jsonify({"error": "Analysis not found"}), 404

    analysis_dict = dict(analysis)
    disease_info = get_disease_info(analysis_dict["disease_class"])

    export_data = {
        "analysis": analysis_dict,
        "disease_info": disease_info
    }

    conn.close()
    return jsonify(export_data)

# API endpoint to export analysis as PDF
@app.route("/api/export/<analysis_id>/pdf")
def export_analysis_pdf(analysis_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    analysis = c.fetchone()

    if not analysis:
        conn.close()
        return jsonify({"error": "Analysis not found"}), 404

    analysis_dict = dict(analysis)
    disease_info = get_disease_info(analysis_dict["disease_class"])

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add content to the PDF
    story.append(Paragraph("Skin Disease Analysis Report", styles['h1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Analysis ID: {analysis_id}", styles['Normal']))
    story.append(Paragraph(f"Disease Class: {analysis_dict['disease_class']}", styles['Normal']))
    story.append(Paragraph(f"Confidence: {analysis_dict['confidence']}", styles['Normal']))

    if disease_info:
        story.append(Paragraph("Disease Information", styles['h2']))
        story.append(Paragraph(f"Description: {disease_info['description']}", styles['Normal']))
        story.append(Paragraph(f"Symptoms: {disease_info['symptoms']}", styles['Normal']))
        story.append(Paragraph(f"Treatments: {disease_info['treatments']}", styles['Normal']))
        story.append(Paragraph(f"Prevention: {disease_info['prevention']}", styles['Normal']))

    doc.build(story)
    buffer.seek(0)

    conn.close()

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'skin_analysis_{analysis_id}.pdf'
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
