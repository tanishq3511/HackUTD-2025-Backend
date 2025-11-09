import os
import base64
import json
import time
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ----------------------------
# Setup environment
# ----------------------------
load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Paths
JSON_PATH = "pdf_summaries.json"
VECTOR_DB_PATH = "./chroma_pdf_db"

# ----------------------------
# PDF → PNG
# ----------------------------
def pdf_to_images(pdf_path: str, output_dir: str, max_pages: int = 4):
    os.makedirs(output_dir, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=200)
    image_paths = []
    for i, page in enumerate(pages[:max_pages]):
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        page.save(image_path, "PNG")
        image_paths.append(image_path)
        print(f"✅ Saved {image_path}")
    return image_paths

# ----------------------------
# Encode Image
# ----------------------------
def encode_image(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# ----------------------------
# Send to NVIDIA Model
# ----------------------------
def send_to_nvidia_model(image_paths, prompt_text: str):
    print(f"🧠 Sending {len(image_paths)} image(s) to Nemotron with prompt: {prompt_text}\n")
    image_messages = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"}}
        for path in image_paths[:4]
    ]

    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        messages=[{"role": "user", "content": image_messages + [{"type": "text", "text": prompt_text}]}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        stream=True
    )

    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            print(piece, end="")
            response_text += piece
    print("\n")
    return response_text

# ----------------------------
# Summarize PDF Pages
# ----------------------------
def summarize_pdf_images(image_paths, prompt_text: str):
    summaries = []
    for i, image_path in enumerate(image_paths):
        print(f"📝 Summarizing page {i+1}...")
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }],
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )
        summary = completion.choices[0].message.content.strip()
        print(f"✅ Page {i+1} Summary:\n{summary}\n")
        summaries.append(summary)
    return summaries

# ----------------------------
# Save summaries → JSON
# ----------------------------
def update_json(folder: str, summaries: list):
    data = {}
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ JSON corrupted — recreating.")
            data = {}
    if "pdf_summaries" not in data or not isinstance(data["pdf_summaries"], list):
        data["pdf_summaries"] = []
    for i, summary in enumerate(summaries):
        entry = {"folder": folder, "page_number": i + 1, "summary": summary}
        data["pdf_summaries"].append(entry)
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved {len(summaries)} summaries to JSON.\n")

# ----------------------------
# Update Vector Store
# ----------------------------
def update_vector_store():
    if not os.path.exists(JSON_PATH):
        return
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07",
        task_type="RETRIEVAL_DOCUMENT"
    )
    vector_store = Chroma(
        collection_name="PDF_Summaries",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    documents, ids = [], []
    for item in data.get("pdf_summaries", []):
        folder, page_number, summary = item["folder"], item["page_number"], item["summary"]
        doc_id = f"{folder}_page_{page_number}"
        documents.append(Document(
            page_content=f"Folder: {folder} | Page: {page_number} | Summary: {summary}",
            metadata={"folder": folder, "page_number": page_number},
            id=doc_id
        ))
        ids.append(doc_id)

    existing_ids = set(vector_store.get()["ids"])
    new_docs = [d for d, id_ in zip(documents, ids) if id_ not in existing_ids]
    new_ids = [id_ for id_ in ids if id_ not in existing_ids]

    if new_docs:
        vector_store.add_documents(documents=new_docs, ids=new_ids)
        print(f"📚 Added {len(new_docs)} new docs to vector DB.\n")
    else:
        print("⚙️ Vector DB already up to date.\n")

# ----------------------------
# Select Relevant Folders for Query
# ----------------------------
def get_relevant_folders(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-exp-03-07",
        task_type="RETRIEVAL_DOCUMENT"
    )
    vector_store = Chroma(
        collection_name="PDF_Summaries",
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    results = vector_store.similarity_search(user_query, k=3)
    folders = set()
    for doc in results:
        folders.add(doc.metadata["folder"])
    return list(folders)

def load_images_from_folders(folders):
    image_paths = []
    for folder in folders:
        folder_path = os.path.join("outputs", folder)
        if os.path.exists(folder_path):
            folder_images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")])
            image_paths.extend(folder_images)
    return image_paths

# ----------------------------
# Master PDF Processor
# ----------------------------
def process_pdf(pdf_path: str, summary_prompt: str):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = int(time.time())
    folder_name = f"{base_name}_{timestamp}"
    output_dir = os.path.join("outputs", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = pdf_to_images(pdf_path, output_dir)
    summaries = summarize_pdf_images(image_paths, summary_prompt)
    update_json(folder_name, summaries)
    update_vector_store()
    return image_paths, folder_name