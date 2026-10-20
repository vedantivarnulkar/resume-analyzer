# ⚡ AI Talent Alignment Engine

An enterprise-grade, multi-page recruitment and candidate-ranking platform that automates resume screening using a robust **4-Factor Scoring Architecture** combining deterministic NLP, semantic vector embeddings, and generative AI insights[cite: 1, 3].

---

## 🏗️ System Architecture & 4-Factor Engine

The application is engineered into four decoupled layers to ensure high performance, modularity, and transparency[cite: 3]:

1. **Layer 1: Input & NLP Processing**[cite: 3]
   * Multi-format document parsing (`PyPDF2`, `Tesseract OCR`, plain text)[cite: 3].
   * Deterministic tokenization and skill extraction via `spaCy PhraseMatcher`[cite: 1, 3].
2. **Layer 2: Hybrid Scoring Engine**[cite: 3]
   * **Skill Match (50%):** Exact keyword extraction paired with `SentenceTransformer` (`all-MiniLM-L6-v2`) semantic similarity vectors to catch contextual equivalents[cite: 3].
   * **Text Similarity (20%):** Normalized cosine similarity between overall document embeddings[cite: 3].
   * **Experience Signals (20%):** Keyword-based seniority and role detection[cite: 3].
   * **Education Signals (10%):** Academic degree and background validation[cite: 3].
3. **Layer 3: Interpretive GenAI Layer (Gemini 1.5 Flash)**[cite: 1, 3]
   * Generates structured markdown evaluations including **Gap Analysis**, **Candidate Strengths**, and **Targeted Interview Probes**.
   * Interactive **AI Chat Panel** allowing recruiters to ask custom follow-up questions about specific candidates.
4. **Layer 4: Frontend UI/UX (Streamlit)**[cite: 3]
   * High-contrast enterprise dark-mode UI with clean state management[cite: 3].
   * Multi-page architecture separating candidate analysis, leaderboard standings, and system configurations[cite: 3].

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Responsive multi-page UI |
| **NLP & Matching** | spaCy, Scikit-Learn, Sentence-Transformers | Phrase matching & cosine similarity vectorization[cite: 3] |
| **OCR & Parsing** | PyPDF2, Tesseract, Pillow | Multi-format resume and document parsing[cite: 3] |
| **Generative AI** | Google Gemini (`gemini-1.5-flash`) | Contextual feedback, gap analysis, and interactive chat[cite: 1, 3] |
| **Version Control** | Git & GitHub | Codebase tracking and remote collaboration[cite: 1, 3] |

---

## 🚀 Getting Started & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/vedantivarnulkar/resume-analyzer.git](https://github.com/vedantivarnulkar/resume-analyzer.git)
cd resume-analyzer
