from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import pdf_conversion

app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "pdfs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

last_uploaded_folder = None
last_image_paths = []


# ----------------------------
# Upload PDF → Process → Summarize
# ----------------------------
@app.route('/api/upload', methods=['POST'])
def upload():
    global last_uploaded_folder, last_image_paths

    if 'file' not in request.files:
        return jsonify({"message": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No filename provided"}), 400

    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Trim path if needed
    if "backend" in file_path:
        trimmed_path = file_path.split("backend" + os.sep, 1)[1]
    else:
        trimmed_path = file_path

    print(f"📄 Received and saved PDF: {file.filename}")
    print(f"📁 Trimmed path: {trimmed_path}")

    try:
        summary_prompt = "Summarize the key information from this document page."
        image_paths, folder_name = pdf_conversion.process_pdf(file_path, summary_prompt)

        last_uploaded_folder = folder_name
        last_image_paths = image_paths

        return jsonify({
            "success": True,
            "message": f"File '{file.filename}' uploaded and processed successfully.",
            "file_name": file.filename,
            "folder_name": folder_name,
            "page_count": len(image_paths)
        }), 200

    except Exception as e:
        print("❌ Error processing PDF:", e)
        return jsonify({
            "success": False,
            "message": f"Error processing '{file.filename}': {str(e)}"
        }), 500


# ----------------------------
# Summarize Most Recent Upload
# ----------------------------
@app.route('/api/summary', methods=['POST'])
def summary():
    global last_image_paths

    if not last_image_paths:
        return jsonify({
            "success": False,
            "message": "No PDF uploaded yet."
        }), 400

    response = pdf_conversion.send_to_nvidia_model(
        last_image_paths,
        "Please summarize this document in clear and concise language."
    )

    return jsonify({
        "success": True,
        "summary": response
    }), 200


# ----------------------------
# Query Stored PDF Summaries
# ----------------------------
@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"success": False, "message": "Missing 'query' parameter"}), 400

    user_query = data["query"]
    print(f"🔍 Query received: {user_query}")

    relevant_folders = pdf_conversion.get_relevant_folders(user_query)
    if not relevant_folders:
        return jsonify({
            "success": True,
            "message": "No relevant PDFs found.",
            "response": None,
            "matched_folders": []
        }), 200

    images_to_use = pdf_conversion.load_images_from_folders(relevant_folders)
    response = pdf_conversion.send_to_nvidia_model(images_to_use, user_query)
    #response = pdf_conversion.send_to_nvidia_model(images_to_use, "")

    return jsonify({
        "success": True,
        "message": "Query processed successfully.",
        "query": user_query,
        "response": response,
        "matched_folders": relevant_folders
    }), 200


# ----------------------------
# Run Flask App
# ----------------------------
if __name__ == '__main__':
    print("🚀 Multi-modal PDF Assistant server running on http://localhost:5000")
    app.run(debug=True)