"""
Crop Doctor - AI-Powered Plant Disease Detection & Advisory API
Enhanced and maintained for Crop_doctor by MGanesh09.
"""

import os
import json
import logging
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crop_doctor")

# -----------------------------
# Configuration & Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

KNOWLEDGE_BASE_FOLDER = os.getenv("KNOWLEDGE_BASE_FOLDER", str(BASE_DIR / "KnowledgeBase"))
FAISS_INDEX_SUBDIR = os.getenv("FAISS_INDEX_SUBDIR", "faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# Globals for state management
vector_store_instance = None
llm_instance = None
Model = None

# Class names for the 41 trained disease categories
class_names = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm',
    'Becterial Blight in Rice', 'Brownspot', 'Common_Rust', 'Cotton Aphid',
    'Flag Smut', 'Gray_Leaf_Spot', 'Healthy Maize', 'Healthy Wheat',
    'Healthy cotton', 'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane',
    'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy',
    'Tungro', 'Wheat Brown leaf Rust', 'Wheat Stem fly', 'Wheat aphid',
    'Wheat black rust', 'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew',
    'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt', 'Yellow Rust Sugarcane',
    'bacterial_blight in Cotton', 'bollworm on Cotton', 'cotton mealy bug',
    'cotton whitefly', 'maize ear rot', 'maize fall armyworm', 'maize stem borer',
    'pink bollworm in cotton', 'red cotton bug', 'thirps on cotton'
]

# -----------------------------
# Built-in Agronomic Knowledge Fallback
# Ensures 100% reliability even if external Gemini API is offline/not configured
# -----------------------------
FALLBACK_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "Common_Rust": {
        "symptoms": "Oval, cinnamon-brown to dark-brown powdery pustules on both upper and lower leaf surfaces.",
        "causes": "Fungus Puccinia sorghi, favored by cool temperatures (16-25°C) and high relative humidity with dew.",
        "prevention": "Plant resistant hybrid varieties, maintain proper crop spacing, and rotate crops with non-hosts like legumes.",
        "treatment": "Apply strobilurin or triazole fungicides (such as azoxystrobin or propiconazole) if infection appears early."
    },
    "Rice Blast": {
        "symptoms": "Spindle-shaped or diamond-shaped lesions with gray or white centers and brown-red margins on leaves and collars.",
        "causes": "Fungus Magnaporthe oryzae, promoted by excessive nitrogen fertilization, frequent rainfall, and warm overcast days.",
        "prevention": "Use certified blast-resistant cultivars, avoid excessive urea application, and maintain balanced potassium and silicon.",
        "treatment": "Spray tricyclazole (0.6 g/L) or isoprothiolane at early tillering and panicle emergence."
    },
    "Becterial Blight in Rice": {
        "symptoms": "Water-soaked stripes along leaf margins turning yellow-white with wavy edges, eventually wilting the foliage.",
        "causes": "Bacterium Xanthomonas oryzae pv. oryzae, entering via wounds or hydathodes during heavy rains and typhoons.",
        "prevention": "Use resistant seeds, ensure clean irrigation canals, avoid deep submergence, and practice crop rotation.",
        "treatment": "Apply copper oxychloride (2.5 g/L) combined with streptocycline (100 ppm) at first sign of lesions."
    },
    "Brownspot": {
        "symptoms": "Circular to oval dark brown lesions with yellow halo on leaves, glumes, and coleoptiles.",
        "causes": "Fungus Bipolaris oryzae, prevalent in nutrient-deficient (low silicon, potassium) and water-stressed soils.",
        "prevention": "Seed treatment with carbendazim (2 g/kg), balanced NPK fertilization, and proper soil water management.",
        "treatment": "Foliar spray of mancozeb or propiconazole (1 ml/L) during early vegetative and booting stages."
    },
    "Leaf Curl": {
        "symptoms": "Upward or downward curling of leaf blades, thickening of veins, stunted plants, and enations on lower surfaces.",
        "causes": "Cotton Leaf Curl Virus (CLCuV), transmitted persistently by the whitefly (Bemisia tabaci).",
        "prevention": "Plant virus-tolerant hybrids, remove alternate weed hosts (Abutilon, Parthenium), and monitor vector populations.",
        "treatment": "Manage whitefly vectors using systemic insecticides like imidacloprid, thiamethoxam, or neem oil (5 ml/L)."
    },
    "Gray_Leaf_Spot": {
        "symptoms": "Rectangular, tan to grayish lesions strictly delimited between leaf veins on corn leaves.",
        "causes": "Fungus Cercospora zeae-maydis, surviving in crop residue under warm, wet, and humid canopy conditions.",
        "prevention": "Deep tillage to bury infected stubble, plant resistant seed varieties, and follow a two-year crop rotation.",
        "treatment": "Foliar fungicide application (strobilurin/triazole premix) before tassel emergence if lesion threshold is crossed."
    },
    "Wheat Brown leaf Rust": {
        "symptoms": "Small, round orange-brown pustules scattered irregularly across upper leaf blades.",
        "causes": "Fungus Puccinia triticina, favored by mild temperatures (15-22°C) and overnight dew formation.",
        "prevention": "Sow rust-resistant wheat varieties, practice timely sowing, and eliminate volunteer wheat plants.",
        "treatment": "Foliar spray with propiconazole 25% EC (Tilt @ 1 ml/L) upon initial detection of pustules."
    },
    "Wheat black rust": {
        "symptoms": "Elongated, reddish-brown to dark pustules on stems, leaf sheaths, and spikes, rupturing the epidermis.",
        "causes": "Fungus Puccinia graminis f. sp. tritici, thriving in warm humid temperatures (20-30°C).",
        "prevention": "Grow stem rust resistant varieties (e.g., Sr-gene hybrids) and eradicate barberry bushes.",
        "treatment": "Timely spray of tebuconazole or propiconazole @ 0.1% to prevent rapid stem lodging."
    },
    "American Bollworm on Cotton": {
        "symptoms": "Circular holes bored into flower buds and bolls with larval excreta accumulated outside.",
        "causes": "Helicoverpa armigera caterpillar feeding on squares, flowers, and developing cotton bolls.",
        "prevention": "Install pheromone traps (5/ha) for monitoring, plant marigold trap crops, and avoid excessive synthetic pyrethroids.",
        "treatment": "Spray NPV (Nuclear Polyhedrosis Virus) or Bacillus thuringiensis (Bt), or emamectin benzoate (0.5 g/L)."
    },
    "cotton mealy bug": {
        "symptoms": "White waxy cotton-like clusters on growing tips, leaves, and bolls; sticky honeydew attracting sooty mold.",
        "causes": "Phenacoccus solenopsis, spread via wind, farm implements, and attending ants.",
        "prevention": "Destroy infested weed hosts, preserve ladybird beetles and encyrtid parasitoid wasps.",
        "treatment": "Spray profenofos 50 EC (2 ml/L) or flonicamid with a sticker/spreader or fish oil rosin soap."
    },
    "RedRot sugarcane": {
        "symptoms": "Third or fourth leaf wilts and turns yellow; cane splitting reveals red internal tissue with white cross patches.",
        "causes": "Fungus Colletotrichum falcatum, primarily seed-piece (sett) transmitted and favored by waterlogging.",
        "prevention": "Plant healthy, disease-free seed setts, treat setts with hot water (52°C for 30 min) or carbendazim, improve drainage.",
        "treatment": "Uproot and burn diseased clumps immediately; apply bioagents like Trichoderma viride to the soil."
    },
    "Healthy cotton": {
        "symptoms": "Vibrant green leaves, sturdy stems, uniform square and boll development with no pathogen lesions.",
        "causes": "Optimal nutrition, clean pest management, and favorable weather conditions.",
        "prevention": "Continue regular pest scouting, balanced irrigation, and micronutrient spray (boron and zinc).",
        "treatment": "No disease treatment required. Maintain current agronomic practices."
    },
    "Healthy Wheat": {
        "symptoms": "Uniform deep green canopy, intact leaf blades, disease-free spikelets and vigorous root growth.",
        "causes": "Balanced nutrient availability and disease-free seed stock.",
        "prevention": "Maintain scheduled irrigation during critical crown root initiation and grain filling stages.",
        "treatment": "No fungicide required. Keep monitoring for aphids and rust."
    },
    "Healthy Maize": {
        "symptoms": "Robust thick stalks, broad dark-green leaves, properly formed ears without lesions or boreholes.",
        "causes": "Proper hybrid selection and fertile, well-aerated soil.",
        "prevention": "Maintain nitrogen split applications and weed-free field borders.",
        "treatment": "No chemical intervention needed."
    },
    "Sugarcane Healthy": {
        "symptoms": "Thick juicy stalks, clean green foliage, absence of reddening, streaks or wilting tops.",
        "causes": "Disease-free sett selection and optimal moisture management.",
        "prevention": "Earthing up on schedule and ensuring proper field drainage.",
        "treatment": "No action needed. Continue standard cultivation cycle."
    }
}

# -----------------------------
# Model Loader
# -----------------------------
def load_keras_model():
    """Safely loads the Keras model from root or saved_models directory."""
    global Model
    candidate_paths = [
        BASE_DIR / "1.keras",
        BASE_DIR / "saved_models" / "1.keras",
        BASE_DIR / "models" / "1.keras"
    ]
    
    model_file = None
    for path in candidate_paths:
        if path.exists():
            model_file = path
            break
            
    if not model_file:
        logger.warning("1.keras model file not found in repository root or saved_models.")
        return None

    try:
        import keras
        logger.info(f"Loading Keras model from {model_file}...")
        Model = keras.models.load_model(str(model_file))
        logger.info("[OK] Keras model loaded successfully.")
        return Model
    except Exception as e:
        logger.error(f"[ERROR] Failed to load Keras model: {e}")
        return None


# -----------------------------
# RAG Helper Functions
# -----------------------------
def get_or_create_vectorstore(knowledge_base_folder: str, embeddings_model: str = EMBEDDINGS_MODEL):
    """Loads existing FAISS index or creates a new one from PDFs."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY not configured. Chatbot will use built-in agronomy knowledge base.")
        return None

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model, google_api_key=google_api_key)
        
        # Check standard path and subdirectory
        possible_paths = [
            os.path.join(knowledge_base_folder, FAISS_INDEX_SUBDIR),
            knowledge_base_folder
        ]
        
        for vs_path in possible_paths:
            faiss_index_file = os.path.join(vs_path, "index.faiss")
            faiss_pkl_file = os.path.join(vs_path, "index.pkl")
            if os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file):
                try:
                    logger.info(f"Loading existing FAISS index from {vs_path}...")
                    return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
                except Exception as e:
                    logger.warning(f"Error loading FAISS index from {vs_path}: {e}")

        # If existing index couldn't be loaded, check for PDFs to build
        if os.path.exists(knowledge_base_folder):
            pdf_files = [f for f in os.listdir(knowledge_base_folder) if f.endswith(".pdf")]
            if pdf_files:
                logger.info(f"Building new FAISS index from {len(pdf_files)} PDF files...")
                all_docs = []
                for f in pdf_files:
                    try:
                        loader = PyPDFLoader(os.path.join(knowledge_base_folder, f))
                        all_docs.extend(loader.load())
                    except Exception as err:
                        logger.warning(f"Failed loading PDF {f}: {err}")
                if all_docs:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(all_docs)
                    target_save = os.path.join(knowledge_base_folder, FAISS_INDEX_SUBDIR)
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    os.makedirs(target_save, exist_ok=True)
                    vector_store.save_local(target_save)
                    logger.info("✅ FAISS index built and saved.")
                    return vector_store

        logger.info("No documents to index. RAG vector store inactive.")
        return None
    except Exception as e:
        logger.warning(f"Could not initialize vector store: {e}")
        return None


def ask_question(query_text: str) -> str:
    """Ask Gemini model a question using RAG and return plain text."""
    if llm_instance is None:
        raise RuntimeError("LLM not initialized.")

    context_text = ""
    if vector_store_instance is not None:
        try:
            docs = vector_store_instance.similarity_search(query_text, k=4)
            context_text = "\n".join([doc.page_content for doc in docs]) if docs else ""
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")

    prompt = f"""You are a professional Agronomist and Plant Pathology Assistant for farmers.
Answer in clear, practical, bulleted language with actionable advice.

Context:
{context_text}

Question:
{query_text}

Answer:"""
    response = llm_instance.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def get_structured_disease_info(query_text: str) -> Dict[str, Any]:
    """Generates structured diagnosis and advice using LLM RAG or offline knowledge base."""
    clean_query = query_text.strip()
    
    # Check greetings
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "help"]
    if clean_query.lower() in greetings:
        msg = "👋 Hello! I am your Crop Doctor assistant. Upload a photo of a crop leaf or ask me about any plant disease (e.g., Rice Blast, Cotton Leaf Curl, Wheat Rust) to get instant symptoms, causes, and treatments!"
        return {
            "type": "GREETING",
            "message": msg,
            "chat_response": msg
        }

    # If LLM is available, use dynamic generative answers
    if llm_instance is not None:
        try:
            triage_prompt = f"Classify the following query as 'GREETING' or 'DISEASE_QUERY': \"{clean_query}\""
            triage_res = llm_instance.invoke(triage_prompt)
            triage_text = triage_res.content.strip().upper() if hasattr(triage_res, "content") else str(triage_res)
            
            if "GREETING" in triage_text:
                msg = "👋 Hello! I am your Crop Doctor assistant. What plant disease or symptom are you noticing today?"
                return {"type": "GREETING", "message": msg, "chat_response": msg}

            def safe_ask(q):
                try:
                    return ask_question(q).strip()
                except Exception:
                    return f"Information currently unavailable for {clean_query}."

            symptoms = safe_ask(f"What are the visible symptoms of {clean_query}?")
            causes = safe_ask(f"What causes {clean_query}?")
            prevention = safe_ask(f"How can farmers prevent {clean_query}?")
            treatment = safe_ask(f"What are the best chemical and organic treatments for {clean_query}?")

            chatbot_message = (
                f"🩺 **Crop Doctor Diagnosis for {clean_query}**:\n\n"
                f"📌 **Symptoms:**\n{symptoms}\n\n"
                f"🔍 **Causes:**\n{causes}\n\n"
                f"🛡️ **Prevention:**\n{prevention}\n\n"
                f"💊 **Treatment:**\n{treatment}\n\n"
                f"Would you like recommendations on specific pesticide dosages or organic remedies?"
            )
            return {
                "type": "DISEASE_INFO",
                "symptoms": symptoms,
                "causes": causes,
                "prevention": prevention,
                "treatment": treatment,
                "chat_response": chatbot_message
            }
        except Exception as e:
            logger.warning(f"LLM query failed: {e}. Falling back to agronomy database.")

    # Offline / Fallback agronomic knowledge
    normalized_key = None
    for k in FALLBACK_KNOWLEDGE:
        if k.lower() in clean_query.lower() or clean_query.lower() in k.lower():
            normalized_key = k
            break

    if normalized_key and normalized_key in FALLBACK_KNOWLEDGE:
        info = FALLBACK_KNOWLEDGE[normalized_key]
        symptoms = info["symptoms"]
        causes = info["causes"]
        prevention = info["prevention"]
        treatment = info["treatment"]
    else:
        symptoms = f"Visible chlorosis, spots, necrosis, or pest damage matching {clean_query}."
        causes = "Biotic stress caused by fungal, bacterial, viral, or insect pest infestation."
        prevention = "Employ certified resistant seeds, balanced fertilizer application, crop rotation, and regular field scouting."
        treatment = "Remove severely infested plant parts. Apply recommended targeted fungicides or bio-pesticides (such as neem oil 5ml/L or copper oxychloride) based on local agronomy guidelines."

    chatbot_message = (
        f"🩺 **Crop Doctor Advice for {clean_query}**:\n\n"
        f"📌 **Symptoms:** {symptoms}\n\n"
        f"🔍 **Causes:** {causes}\n\n"
        f"🛡️ **Prevention:** {prevention}\n\n"
        f"💊 **Treatment:** {treatment}\n"
    )

    return {
        "type": "DISEASE_INFO",
        "symptoms": symptoms,
        "causes": causes,
        "prevention": prevention,
        "treatment": treatment,
        "chat_response": chatbot_message
    }


# -----------------------------
# Image Helper Functions
# -----------------------------
def read_file_as_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)).convert("RGB"))


def predict_image(image: np.ndarray):
    """Processes image and runs CNN inference."""
    global Model
    if Model is None:
        Model = load_keras_model()
        if Model is None:
            raise HTTPException(status_code=503, detail="Trained model '1.keras' is not loaded.")

    img_pil = Image.fromarray(image).resize((128, 128))
    img_array = np.array(img_pil).astype("float32")
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = Model.predict(img_batch, verbose=0)
    pred_idx = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    
    if pred_idx < len(class_names):
        predicted_class = class_names[pred_idx]
    else:
        predicted_class = f"Class_{pred_idx}"

    return predicted_class, confidence


# -----------------------------
# FastAPI Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup (loading models/index) and shutdown gracefully."""
    global vector_store_instance, llm_instance
    logger.info("Initializing Crop Doctor API...")

    # Load Keras CNN Model
    load_keras_model()

    # Load Gemini LLM and Vector Store if API key available
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key and google_api_key != "your_gemini_api_key_here":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm_instance = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE, google_api_key=google_api_key)
            vector_store_instance = get_or_create_vectorstore(KNOWLEDGE_BASE_FOLDER)
            logger.info("[OK] Gemini LLM & RAG vector store initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load Gemini LLM: {e}. Fallback knowledge enabled.")
    else:
        logger.info("[INFO] GOOGLE_API_KEY not configured. Running with high-speed built-in agronomy database.")

    yield
    logger.info("Crop Doctor API shutdown complete.")


# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(
    title="Crop Doctor API",
    description="AI-powered plant disease detection and agricultural advisory system",
    version="2.2.0",
    lifespan=lifespan
)

# CORS Middleware setup - Supports frontend web clients & localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request & Response Schemas
# -----------------------------
class DiseaseQuery(BaseModel):
    disease_name: str


# -----------------------------
# Interactive Web Dashboard
# -----------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Crop Doctor - AI Plant Disease Detection</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1310;
      --card-bg: rgba(18, 32, 26, 0.85);
      --primary: #10b981;
      --primary-hover: #059669;
      --accent: #34d399;
      --border: rgba(52, 211, 153, 0.2);
      --text: #f0fdf4;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
    body {
      background: radial-gradient(circle at 50% 0%, #163628 0%, var(--bg) 80%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }
    .header { text-align: center; margin-bottom: 2rem; }
    .header h1 { font-size: 2.5rem; color: #6ee7b7; font-weight: 700; margin-bottom: 0.5rem; }
    .header p { color: var(--text-muted); font-size: 1.1rem; }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent);
      border: 1px solid var(--border);
      border-radius: 9999px;
      font-size: 0.85rem;
      margin-top: 0.5rem;
    }
    .container {
      width: 100%;
      max-width: 900px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }
    @media (max-width: 768px) { .container { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .card h2 { font-size: 1.3rem; margin-bottom: 1rem; color: #a7f3d0; }
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 2rem 1rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s ease;
      background: rgba(16, 185, 129, 0.03);
    }
    .dropzone:hover {
      border-color: var(--accent);
      background: rgba(16, 185, 129, 0.08);
    }
    .preview-box {
      margin-top: 1rem;
      text-align: center;
      display: none;
    }
    .preview-box img {
      max-width: 100%;
      max-height: 200px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .btn {
      width: 100%;
      margin-top: 1rem;
      padding: 0.85rem 1.5rem;
      background: linear-gradient(135deg, var(--primary), var(--primary-hover));
      color: #fff;
      border: none;
      border-radius: 10px;
      font-weight: 600;
      font-size: 1rem;
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .btn:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    .chat-input {
      display: flex;
      gap: 0.5rem;
      margin-top: 1rem;
    }
    .chat-input input {
      flex: 1;
      padding: 0.75rem 1rem;
      background: rgba(0,0,0,0.3);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-size: 0.95rem;
    }
    .chat-input button {
      padding: 0.75rem 1.25rem;
      background: var(--primary);
      border: none;
      border-radius: 8px;
      color: #fff;
      font-weight: 600;
      cursor: pointer;
    }
    .result-section {
      margin-top: 1rem;
      padding: 1rem;
      background: rgba(0, 0, 0, 0.35);
      border-radius: 10px;
      border-left: 4px solid var(--primary);
      display: none;
    }
    .result-title { font-size: 1.15rem; font-weight: 700; color: #6ee7b7; margin-bottom: 0.5rem; }
    .badge-conf { background: #064e3b; color: #6ee7b7; padding: 0.2rem 0.5rem; border-radius: 6px; font-size: 0.85rem; }
    .info-block { margin-top: 0.75rem; font-size: 0.9rem; line-height: 1.5; color: #cbd5e1; }
    .info-block strong { color: #a7f3d0; }
    .links { margin-top: 2rem; display: flex; gap: 1rem; font-size: 0.9rem; }
    .links a { color: var(--accent); text-decoration: none; }
    .links a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="header">
    <h1>🌱 Crop Doctor</h1>
    <p>AI-Powered Plant Disease Detection & Agricultural Advisory</p>
    <div class="badge">Repository: MGanesh09 / Crop_doctor &bull; Model: CNN 1.keras</div>
  </div>

  <div class="container">
    <!-- Image Diagnostic Card -->
    <div class="card">
      <h2>📸 Leaf Image Diagnosis</h2>
      <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <p>🌿 Drag & drop plant leaf image or <strong>browse</strong></p>
        <span style="font-size: 0.8rem; color: var(--text-muted)">Supports JPG, PNG, WEBP</span>
      </div>
      <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFile(this.files[0])">
      <div class="preview-box" id="previewBox">
        <img id="previewImg" alt="Preview">
      </div>
      <button class="btn" id="analyzeBtn" onclick="analyzeImage()" disabled>Diagnose Disease</button>
      
      <div class="result-section" id="imgResult">
        <div class="result-title" id="predClass">Result</div>
        <div>Confidence: <span class="badge-conf" id="predConf">0%</span></div>
        <div class="info-block" id="predAdvice"></div>
      </div>
    </div>

    <!-- Chatbot Advisory Card -->
    <div class="card">
      <h2>💬 Crop Doctor Advisory</h2>
      <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.5rem;">
        Ask about remedies, symptoms, or preventive care for any crop disease:
      </p>
      <div class="chat-input">
        <input type="text" id="chatQuery" placeholder="e.g. Rice Blast treatment or Cotton Bollworm" onkeypress="if(event.key==='Enter')askChatbot()">
        <button onclick="askChatbot()">Ask</button>
      </div>

      <div class="result-section" id="chatResult">
        <div class="result-title" id="chatTitle">Advisory</div>
        <div class="info-block" id="chatBody"></div>
      </div>
    </div>
  </div>

  <div class="links">
    <a href="/docs" target="_blank">📖 Swagger API Docs</a>
    <a href="/health" target="_blank">🩺 Health Status</a>
    <a href="https://github.com/MGanesh09/Crop_doctor" target="_blank">⭐ GitHub Repository</a>
  </div>

  <script>
    let selectedFile = null;
    function handleFile(file) {
      if (!file) return;
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = e => {
        document.getElementById('previewImg').src = e.target.result;
        document.getElementById('previewBox').style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;
      };
      reader.readAsDataURL(file);
    }

    async function analyzeImage() {
      if (!selectedFile) return;
      const btn = document.getElementById('analyzeBtn');
      btn.innerText = 'Analyzing Leaf...';
      btn.disabled = true;

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const res = await fetch('/analyze_disease_from_image', { method: 'POST', body: formData });
        const data = await res.json();
        
        document.getElementById('predClass').innerText = data.predicted_class || 'Unknown';
        document.getElementById('predConf').innerText = ((data.confidence || 0) * 100).toFixed(1) + '%';
        
        const ans = data.chatbot_answer || {};
        let adviceHtml = '';
        if (ans.symptoms) adviceHtml += `<p><strong>Symptoms:</strong> ${ans.symptoms}</p><br>`;
        if (ans.treatment) adviceHtml += `<p><strong>Treatment:</strong> ${ans.treatment}</p><br>`;
        if (ans.prevention) adviceHtml += `<p><strong>Prevention:</strong> ${ans.prevention}</p>`;
        if (!adviceHtml && ans.chat_response) adviceHtml = ans.chat_response.replace(/\\n/g, '<br>');
        
        document.getElementById('predAdvice').innerHTML = adviceHtml || 'Diagnosis complete.';
        document.getElementById('imgResult').style.display = 'block';
      } catch (err) {
        alert('Analysis error: ' + err.message);
      } finally {
        btn.innerText = 'Diagnose Disease';
        btn.disabled = false;
      }
    }

    async function askChatbot() {
      const q = document.getElementById('chatQuery').value.trim();
      if (!q) return;
      const resSection = document.getElementById('chatResult');
      document.getElementById('chatTitle').innerText = 'Consulting Crop Doctor...';
      document.getElementById('chatBody').innerText = 'Fetching treatment & agronomy guide...';
      resSection.style.display = 'block';

      try {
        const res = await fetch('/chatbot_disease_info', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ disease_name: q })
        });
        const data = await res.json();
        document.getElementById('chatTitle').innerText = data.disease_name || 'Crop Doctor Advice';
        
        if (data.structured_data) {
          const sd = data.structured_data;
          document.getElementById('chatBody').innerHTML = `
            <p><strong>Symptoms:</strong> ${sd.symptoms}</p><br>
            <p><strong>Causes:</strong> ${sd.causes}</p><br>
            <p><strong>Prevention:</strong> ${sd.prevention}</p><br>
            <p><strong>Treatment:</strong> ${sd.treatment}</p>
          `;
        } else {
          document.getElementById('chatBody').innerHTML = (data.chatbot_message || 'No advice found.').replace(/\\n/g, '<br>');
        }
      } catch (err) {
        document.getElementById('chatBody').innerText = 'Error: ' + err.message;
      }
    }
  </script>
</body>
</html>
"""


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serves the interactive dashboard for browser visitors, and JSON for API clients."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=HTML_PAGE)
    return JSONResponse({
        "message": "Crop Doctor API is running!",
        "author": "MGanesh09",
        "docs_url": "/docs",
        "ui_url": "/",
        "model_loaded": Model is not None,
        "llm_active": llm_instance is not None
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Crop Doctor API",
        "version": "2.2.0",
        "model_loaded": Model is not None,
        "llm_loaded": llm_instance is not None,
        "vector_store_loaded": vector_store_instance is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict crop disease from uploaded image using local CNN model."""
    try:
        content = await file.read()
        image = read_file_as_image(content)
        predicted_class, confidence = predict_image(image)
        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {str(e)}")


@app.post("/analyze_disease_from_image")
async def analyze_disease(file: UploadFile = File(...)):
    """Predict disease from image and retrieve comprehensive agronomic advice."""
    try:
        content = await file.read()
        image = read_file_as_image(content)
        predicted_class, confidence = predict_image(image)

        logger.info(f"Detected: {predicted_class} (confidence: {confidence:.2f})")
        disease_info = get_structured_disease_info(predicted_class)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "chatbot_answer": disease_info
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Disease analysis failed: {str(e)}")


@app.post("/chatbot_disease_info")
async def chatbot_disease_info(query: DiseaseQuery):
    """Retrieve structured disease information and conversational advice."""
    if not query.disease_name:
        raise HTTPException(status_code=400, detail="Disease name is required.")

    logger.info(f"Chatbot query: {query.disease_name}")
    disease_info = get_structured_disease_info(query.disease_name)

    if disease_info["type"] == "GREETING":
        return {
            "query_type": "greeting",
            "chatbot_message": disease_info["chat_response"]
        }

    return {
        "query_type": "disease_info",
        "disease_name": query.disease_name,
        "chatbot_message": disease_info["chat_response"],
        "structured_data": {
            "symptoms": disease_info.get("symptoms", ""),
            "causes": disease_info.get("causes", ""),
            "prevention": disease_info.get("prevention", ""),
            "treatment": disease_info.get("treatment", "")
        }
    }


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "9087"))
    print(f"[*] Starting Crop Doctor API on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)