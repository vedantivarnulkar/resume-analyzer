# ==============================================================================
# requirements.txt
# ==============================================================================
# scikit-learn>=1.3.0
# PyPDF2>=3.0.0
# python-docx>=1.0.0
# ==============================================================================

"""
USAGE GUIDE:
------------
1. Install the required third-party packages listed in the requirements section above:
   pip install scikit-learn PyPDF2 python-docx

2. Import the `BatchResumeRanker` class into your recruitment platform service,
   or configure the placeholders (`[RESUME_DIRECTORY]`, `[JOB_DESCRIPTION]`,
   `[OUTPUT_PATH]`) in the `__main__` block at the bottom of this file.

3. Call `rank_resumes_from_directory(resume_dir, job_description, top_k=5)` to
   process a batch of resumes and retrieve the top 5 matches in descending order.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CandidateMatch:
    """Data structure representing a ranked candidate resume."""
    rank: int
    resume_identifier: str
    file_path: str
    similarity_score: float


class ResumeParser:
    """Handles text extraction and normalization across multiple file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    @classmethod
    def extract_text(cls, file_path: Path) -> str:
        """
        Extracts raw text from a resume file based on its extension.
        Returns an empty string if the file cannot be read or is unsupported.
        """
        suffix = file_path.suffix.lower()
        if suffix not in cls.SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file format: %s", file_path.name)
            return ""

        try:
            if suffix == ".pdf":
                return cls._extract_from_pdf(file_path)
            elif suffix == ".docx":
                return cls._extract_from_docx(file_path)
            elif suffix == ".txt":
                return cls._extract_from_txt(file_path)
        except Exception as exc:
            logger.error("Failed to extract text from %s: %s", file_path.name, exc)
            return ""

        return ""

    @staticmethod
    def _extract_from_pdf(file_path: Path) -> str:
        """Extracts text from all pages of a PDF document."""
        text_chunks: List[str] = []
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
        return "\n".join(text_chunks)

    @staticmethod
    def _extract_from_docx(file_path: Path) -> str:
        """Extracts text from paragraphs and tables in a DOCX document."""
        doc = docx.Document(file_path)
        chunks: List[str] = [p.text for p in doc.paragraphs if p.text]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        chunks.append(cell.text)
        return "\n".join(chunks)

    @staticmethod
    def _extract_from_txt(file_path: Path) -> str:
        """Reads plain-text resume files safely."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as txt_file:
            return txt_file.read()

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Normalizes whitespace and strips non-printable characters."""
        if not raw_text:
            return ""
        # Replace control characters and collapse multiple spaces/newlines
        cleaned = re.sub(r"[\r\n\t]+", " ", raw_text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()


class BatchResumeRanker:
    """
    Computes NLP similarity scores between a job description and a batch of resumes,
    returning the top K most relevant candidates.
    """

    def __init__(self, ngram_range: tuple = (1, 2), max_features: Optional[int] = 10000):
        """
        Initializes the TF-IDF vectorizer with English stop-word removal
        and n-gram support to capture both single keywords and multi-word phrases.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            sublinear_tf=True
        )

    def load_resumes(self, resume_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Scans the target directory and extracts cleaned text from all valid resumes.
        Returns a dictionary mapping resume identifiers to file paths and extracted text.
        """
        dir_path = Path(resume_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

        parsed_resumes: Dict[str, Dict[str, str]] = {}

        for file_path in sorted(dir_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in ResumeParser.SUPPORTED_EXTENSIONS:
                raw_text = ResumeParser.extract_text(file_path)
                cleaned_text = ResumeParser.clean_text(raw_text)

                if cleaned_text:
                    parsed_resumes[file_path.name] = {
                        "file_path": str(file_path.resolve()),
                        "text": cleaned_text
                    }
                else:
                    logger.warning("No readable text extracted from: %s", file_path.name)

        return parsed_resumes

    def score_and_rank(
        self,
        resumes: Dict[str, Dict[str, str]],
        job_description: str,
        top_k: int = 5
    ) -> List[CandidateMatch]:
        """
        Vectorizes the job description and resume texts, calculates cosine similarity,
        and returns the top `top_k` candidates in descending order of relevance.
        """
        cleaned_jd = ResumeParser.clean_text(job_description)
        if not cleaned_jd:
            raise ValueError("The provided job description is empty after text normalization.")

        if not resumes:
            logger.warning("No valid resumes provided for ranking.")
            return []

        identifiers = list(resumes.keys())
        resume_texts = [resumes[ident]["text"] for ident in identifiers]

        # Index 0 is the Job Description; indices 1..N are the candidate resumes
        corpus = [cleaned_jd] + resume_texts
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Compute cosine similarity between the JD vector (row 0) and all resume vectors (rows 1..N)
        jd_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]
        similarity_scores = cosine_similarity(jd_vector, resume_vectors).flatten()

        # Pair resumes with their similarity scores and sort descending
        scored_indices = sorted(
            enumerate(similarity_scores),
            key=lambda item: item[1],
            reverse=True
        )

        # Select the top K matches
        top_matches: List[CandidateMatch] = []
        for rank_idx, (original_idx, score) in enumerate(scored_indices[:top_k], start=1):
            ident = identifiers[original_idx]
            top_matches.append(
                CandidateMatch(
                    rank=rank_idx,
                    resume_identifier=ident,
                    file_path=resumes[ident]["file_path"],
                    similarity_score=round(float(score), 4)
                )
            )

        return top_matches

    def rank_resumes_from_directory(
        self,
        resume_dir: str,
        job_description: str,
        top_k: int = 5,
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        End-to-end pipeline: loads resumes from a directory, ranks them against
        the job description, optionally writes results to disk, and returns the top K list.
        """
        resumes = self.load_resumes(resume_dir)
        top_candidates = self.score_and_rank(resumes, job_description, top_k=top_k)
        serialized_results = [asdict(candidate) for candidate in top_candidates]

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(serialized_results, f, indent=4)
            logger.info("Saved top %d candidate rankings to %s", len(serialized_results), output_path)

        return serialized_results


# ==============================================================================
# Example Invocation
# ==============================================================================
if __name__ == "__main__":
    # Replace bracketed placeholders with actual runtime configurations
    RESUME_DIR_INPUT = "[RESUME_DIRECTORY]"
    JOB_DESCRIPTION_INPUT = "[JOB_DESCRIPTION]"
    OUTPUT_FILE_PATH = "[OUTPUT_PATH]"
    TOP_K_CANDIDATES = 5

    ranker = BatchResumeRanker(ngram_range=(1, 2))

    try:
        top_5_matches = ranker.rank_resumes_from_directory(
            resume_dir=RESUME_DIR_INPUT,
            job_description=JOB_DESCRIPTION_INPUT,
            top_k=TOP_K_CANDIDATES,
            output_path=OUTPUT_FILE_PATH
        )

        for match in top_5_matches:
            print(
                f"Rank #{match['rank']}: {match['resume_identifier']} "
                f"(Similarity Score: {match['similarity_score']:.4f})"
            )
    except Exception as error:
        logger.error("Batch ranking execution halted: %s", error)



import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
import json
import PyPDF2
from PIL import Image
import pytesseract
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Talent Alignment Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Dark Mode & Professional UI/UX
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Inter', -apple-system, sans-serif; color: #F8F9FA !important; font-weight: 700; }
    .metric-card {
        background: #1A1C24; border: 1px solid #2D3748; border-radius: 12px;
        padding: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: #3B82F6; transform: translateY(-4px); }
    .metric-card h4 { color: #94A3B8 !important; font-size: 0.9rem; margin-bottom: 8px; text-transform: uppercase; }
    
    .badge-found {
        display: inline-block; background-color: rgba(16, 185, 129, 0.15); color: #34D399;
        border: 1px solid #059669; padding: 6px 12px; margin: 4px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;
    }
    .badge-semantic {
        display: inline-block; background-color: rgba(59, 130, 246, 0.15); color: #60A5FA;
        border: 1px solid #3B82F6; padding: 6px 12px; margin: 4px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;
    }
    .badge-missing {
        display: inline-block; background-color: rgba(239, 68, 68, 0.15); color: #F87171;
        border: 1px solid #DC2626; padding: 6px 12px; margin: 4px; border-radius: 6px; font-size: 0.85rem; font-weight: 600;
    }
    .streamlit-expanderHeader { background-color: #1A1C24 !important; color: #F8F9FA !important; border-radius: 8px; border: 1px solid #2D3748; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if "processed_results" not in st.session_state:
    st.session_state.processed_results = None
if "final_jd_text" not in st.session_state:
    st.session_state.final_jd_text = ""
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ---------------------------------------------------------
# Static Admin Weights (Locked from UI changes)
# ---------------------------------------------------------
STATIC_WEIGHTS = {
    "Programming & Languages": 1.20,
    "Frameworks & Infrastructure": 1.00,
    "CS Concepts & Methodologies": 0.80
}

# ---------------------------------------------------------
# Utilities & NLP/Embedding Models
# ---------------------------------------------------------
@st.cache_resource
def load_nlp_pipeline():
    try: return spacy.load("en_core_web_sm")
    except: return spacy.blank("en")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages: text += page.extract_text() + "\n"
        elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
        elif uploaded_file.type == "text/plain":
            text = str(uploaded_file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error parsing file: {e}")
    return text.strip()

DYNAMIC_SKILL_TAXONOMY = {
    "Programming & Languages": ["Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "Go", "Ruby", "SQL", "HTML", "CSS", "Bash", "Swift", "Kotlin"],
    "Frameworks & Infrastructure": ["React", "Node.js", "Django", "Spring Boot", "Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Linux", "CI/CD", "Firebase", "Next.js", "Redux", "Express", "MongoDB", "PostgreSQL"],
    "CS Concepts & Methodologies": ["Data Structures", "Algorithms", "Agile", "Scrum", "Machine Learning", "System Design", "REST API", "Microservices", "Root Cause Analysis", "Authentication", "Authorization", "Testing", "Debugging"]
}

# ---------------------------------------------------------
# ADVANCED SCORING ENGINE (Fixed Math Logic)
# ---------------------------------------------------------
def calculate_comprehensive_score(jd_text, resume_text, weights):
    nlp = load_nlp_pipeline()
    embedder = load_embedding_model()
    
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for cat, terms in DYNAMIC_SKILL_TAXONOMY.items():
        matcher.add(cat, [nlp.make_doc(t) for t in terms])
        
    def extract(text):
        doc = nlp(text)
        extracted = {cat: set() for cat in DYNAMIC_SKILL_TAXONOMY.keys()}
        for match_id, start, end in matcher(doc):
            cat = nlp.vocab.strings[match_id]
            extracted[cat].add(doc[start:end].text.title())
        return extracted

    jd_skills = extract(jd_text)
    res_skills = extract(resume_text)
    
    resume_sentences = [sent.text for sent in nlp(resume_text).sents if len(sent.text.strip()) > 10]
    resume_vectors = embedder.encode(resume_sentences) if resume_sentences else []

    category_scores = {}
    weighted_numerator = 0.0
    active_weight_total = 0.0 # Fix: Only count weights if the JD actually asks for skills in this category

    for cat in jd_skills.keys():
        jd_set = set(jd_skills[cat])
        res_set = set(res_skills[cat])
        
        # If JD doesn't ask for any skills in this category, skip it mathematically so it doesn't inflate scores
        if len(jd_set) == 0:
            category_scores[cat] = {"rate": 0.0, "exact": [], "semantic": [], "missing": [], "jd_count": 0}
            continue
            
        exact = jd_set.intersection(res_set)
        missing = list(jd_set.difference(res_set))
        semantic = []
        
        if len(resume_vectors) > 0 and missing:
            skill_vecs = embedder.encode(missing)
            sim_matrix = cosine_similarity(skill_vecs, resume_vectors)
            for idx, skill in enumerate(missing):
                if np.max(sim_matrix[idx]) >= 0.45: semantic.append(skill)
                    
        final_missing = [s for s in missing if s not in semantic]
        cat_points = len(exact) + (len(semantic) * 0.8)
        cat_rate = min(cat_points / len(jd_set), 1.0)
        
        category_scores[cat] = {
            "rate": cat_rate, "exact": sorted(list(exact)),
            "semantic": sorted(semantic), "missing": sorted(final_missing), "jd_count": len(jd_set)
        }
        
        weighted_numerator += cat_rate * weights.get(cat, 1.0)
        active_weight_total += weights.get(cat, 1.0)
        
    skill_match_score = (weighted_numerator / active_weight_total) * 100 if active_weight_total > 0 else 0

    jd_vec = embedder.encode(jd_text)
    res_vec = embedder.encode(resume_text)
    text_sim = cosine_similarity([jd_vec], [res_vec])[0][0]
    text_similarity_score = max(0, min(100, text_sim * 100))

    exp_keywords = ['experience', 'intern', 'internship', 'developer', 'engineer', 'work', 'project', 'professional', 'role']
    exp_count = sum(1 for kw in exp_keywords if kw in resume_text.lower())
    experience_score = min(100, exp_count * 15)

    edu_keywords = ['bachelor', 'master', 'degree', 'university', 'college', 'b.s.', 'b.a.', 'computer science', 'phd']
    edu_count = sum(1 for kw in edu_keywords if kw in resume_text.lower())
    education_score = min(100, edu_count * 35)

    final_score = (0.50 * skill_match_score) + (0.20 * text_similarity_score) + (0.20 * experience_score) + (0.10 * education_score)
    return round(final_score, 1), round(skill_match_score, 1), round(text_similarity_score, 1), round(experience_score, 1), round(education_score, 1), category_scores

# ---------------------------------------------------------
# LLM Generation & Chat Engine (Using 3.8-flash)
# ---------------------------------------------------------
def generate_llm_feedback(jd_text, resume_text, missing_skills, score):
    genai.configure(api_key=st.session_state.api_key)
    # Using 1.5-flash as it is the most stable and universally accessible endpoint
    model = genai.GenerativeModel('gemini-3.8-flash')
    prompt = f"""
    You are an expert technical hiring manager evaluating a candidate.
    An engine evaluated this resume with an overall {score}% match. Missing skills: {json.dumps(missing_skills)}
    Job Description: {jd_text[:1200]} | Resume: {resume_text[:1200]}
    
    Provide a concise evaluation in Markdown:
    1. **Gap Analysis**: Are the missing skills dealbreakers?
    2. **Candidate Strength**: Their strongest applicable asset.
    3. **Interview Probes**: 2 precise questions regarding their gaps.
    """
    return model.generate_content(prompt).text

def chat_with_ai(jd_text, resume_text, question):
    genai.configure(api_key=st.session_state.api_key)
    model = genai.GenerativeModel('gemini-3.8-flash')
    prompt = f"""
    You are an AI hiring assistant. Answer the user's question about the candidate based on the resume and job description.
    Job Description: {jd_text[:1000]}
    Resume: {resume_text[:1000]}
    User Question: {question}
    Answer professionally and concisely.
    """
    return model.generate_content(prompt).text

# ---------------------------------------------------------
# Main UI App Layout
# ---------------------------------------------------------
st.title("⚡ AI Talent Alignment Engine")
st.markdown("**Powered by Hybrid NLP & Gemini 3.8 Flash.**")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Save API key securely in session state so it doesn't reset
    api_key_input = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password", placeholder="Paste key here...")
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        
    st.subheader("⚖️ Scoring Engine Weights")
    st.info("🔒 **Admin Locked**\n\nThe internal mathematical weights for the skill extraction algorithm have been fixed by the system administrator to maintain standardized scoring across all candidates.")
    st.markdown(f"""
    - **Programming**: {STATIC_WEIGHTS['Programming & Languages']}x
    - **Frameworks**: {STATIC_WEIGHTS['Frameworks & Infrastructure']}x
    - **Methodologies**: {STATIC_WEIGHTS['CS Concepts & Methodologies']}x
    """)

# --- INPUT SECTION ---
st.markdown("### 📥 Input Data")
col_jd, col_res = st.columns(2)
with col_jd:
    jd_file = st.file_uploader("Upload JD (PDF, TXT)", type=["pdf", "txt"])
    st.markdown("**— OR —**")
    jd_text_input = st.text_area("Paste JD Text Directly:", height=150)
with col_res:
    resume_files = st.file_uploader("Upload Resumes (PDF, Image)", type=["pdf", "jpg", "png"], accept_multiple_files=True)
    st.markdown("**— OR —**")
    resume_text_input = st.text_area("Paste Resume Text Directly:", height=150)

# --- EXECUTION ---
if st.button("🚀 Process & Rank Candidates", type="primary", use_container_width=True):
    final_jd_text = extract_text_from_file(jd_file) if jd_file else jd_text_input.strip()

    if not final_jd_text or (not resume_files and not resume_text_input.strip()):
        st.error("⚠️ Please provide both a Job Description and at least one Resume.")
    else:
        with st.spinner("Executing 4-Factor Architecture..."):
            results = []
            
            # If files were uploaded
            if resume_files:
                for res_file in resume_files:
                    res_text = extract_text_from_file(res_file)
                    final_score, skill_sc, text_sc, exp_sc, edu_sc, breakdown = calculate_comprehensive_score(final_jd_text, res_text, STATIC_WEIGHTS)
                    missing = [skill for data in breakdown.values() for skill in data["missing"]]
                    
                    results.append({
                        "name": res_file.name, "text": res_text, "score": final_score,
                        "skill_sc": skill_sc, "text_sc": text_sc, "exp_sc": exp_sc, "edu_sc": edu_sc,
                        "breakdown": breakdown, "missing": missing,
                        "ai_feedback": None, "chat_history": [] 
                    })
            
            # If text was pasted
            if resume_text_input.strip():
                res_text = resume_text_input.strip()
                final_score, skill_sc, text_sc, exp_sc, edu_sc, breakdown = calculate_comprehensive_score(final_jd_text, res_text, STATIC_WEIGHTS)
                missing = [skill for data in breakdown.values() for skill in data["missing"]]
                
                results.append({
                    "name": "Pasted Candidate", "text": res_text, "score": final_score,
                    "skill_sc": skill_sc, "text_sc": text_sc, "exp_sc": exp_sc, "edu_sc": edu_sc,
                    "breakdown": breakdown, "missing": missing,
                    "ai_feedback": None, "chat_history": [] 
                })
                
            st.session_state.processed_results = sorted(results, key=lambda x: x["score"], reverse=True)
            st.session_state.final_jd_text = final_jd_text

# --- RENDER RESULTS ---
if st.session_state.processed_results:
    st.divider()
    st.markdown("### 🏆 Candidate Leaderboard")
    
    for idx, c in enumerate(st.session_state.processed_results):
        fit_color = "#34D399" if c["score"] >= 75 else ("#FBBF24" if c["score"] >= 50 else "#F87171")
        
        with st.expander(f"{idx+1}. {c['name']} — Final Score: {c['score']}%", expanded=(idx==0)):
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: st.markdown(f'<div class="metric-card"><h4>Overall Match</h4><h2 style="color: {fit_color};">{c["score"]}%</h2></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="metric-card"><h4>Skills (50%)</h4><h2 style="color: #60A5FA;">{c["skill_sc"]}%</h2></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="metric-card"><h4>Text Sim (20%)</h4><h2 style="color: #F8F9FA;">{c["text_sc"]}%</h2></div>', unsafe_allow_html=True)
            with m4: st.markdown(f'<div class="metric-card"><h4>Exp (20%)</h4><h2 style="color: #F8F9FA;">{c["exp_sc"]}%</h2></div>', unsafe_allow_html=True)
            with m5: st.markdown(f'<div class="metric-card"><h4>Edu (10%)</h4><h2 style="color: #F8F9FA;">{c["edu_sc"]}%</h2></div>', unsafe_allow_html=True)
                
            st.write("")
            tab_gap, tab_ai = st.tabs(["📊 Phase 1: Skill Gap Matrix", "🤖 Phase 2: AI Assistant & Chat"])
            
            # Tab 1: Math & Gaps
            with tab_gap:
                for cat, data in c["breakdown"].items():
                    if data['jd_count'] > 0:
                        st.markdown(f"**{cat}**")
                        st.progress(data['rate'])
                        c_found, c_missing = st.columns(2)
                        with c_found:
                            if data["exact"]: st.markdown(" ".join([f'<span class="badge-found">✓ {s}</span>' for s in data["exact"]]), unsafe_allow_html=True)
                            if data["semantic"]: st.markdown(" ".join([f'<span class="badge-semantic">~ {s} (Semantic)</span>' for s in data["semantic"]]), unsafe_allow_html=True)
                            if not data["exact"] and not data["semantic"]: st.caption("No matches.")
                        with c_missing:
                            if data["missing"]: st.markdown(" ".join([f'<span class="badge-missing">✗ {s}</span>' for s in data["missing"]]), unsafe_allow_html=True)
                            else: st.caption("All requirements met.")
                        st.write("")
                    else:
                        st.caption(f"*(No {cat} requirements detected in Job Description)*")
            
            # Tab 2: AI Analysis & Interactive Chat
            with tab_ai:
                if not st.session_state.api_key:
                    st.warning("⚠️ Enter your Gemini API Key in the sidebar to unlock AI tools.")
                else:
                    # 1. Generate/Display Base Feedback
                    if not c["ai_feedback"]:
                        if st.button(f"Generate Baseline Analysis for {c['name']}", key=f"btn_{idx}"):
                            with st.spinner("Gemini is analyzing..."):
                                c["ai_feedback"] = generate_llm_feedback(st.session_state.final_jd_text, c["text"], c["missing"], c["score"])
                                st.rerun()
                    else:
                        st.markdown(c["ai_feedback"])
                        st.divider()
                        
                        # 2. Interactive AI Chat System
                        st.markdown(f"#### 💬 Ask AI about {c['name']}")
                        for msg in c["chat_history"]:
                            with st.chat_message(msg["role"]): st.write(msg["content"])
                            
                        # Unique chat input for each candidate
                        if prompt := st.chat_input("Ask a follow-up question...", key=f"chat_{idx}"):
                            c["chat_history"].append({"role": "user", "content": prompt})
                            with st.spinner("AI is thinking..."):
                                try:
                                    answer = chat_with_ai(st.session_state.final_jd_text, c["text"], prompt)
                                    c["chat_history"].append({"role": "assistant", "content": answer})
                                except Exception as e:
                                    st.error(f"Error: {e}")
                            st.rerun()