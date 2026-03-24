# 🧠 Automated Concept Map Generator

Transform unstructured text and PDF documents into visual, interactive concept maps using Natural Language Processing (NLP) and Graph Theory.

## 🚀 Key Features
- **Multi-Source Input**: Support for raw text, PDF, and TXT file uploads.
- **Advanced NLP Extraction**: Combines **Hearst Patterns** (Regex) with **Dependency Parsing** (SpaCy) to identify hierarchical and semantic relationships (hypernyms, hyponyms, and SVO triplets).
- **Interactive Visualizations**: Generates dynamic graphs using **NetworkX** and **Matplotlib**.
- **Mermaid.js Integration**: Automatically creates Mermaid-compatible code for easy sharing and further customization.
- **Global & Local Context**: Provides both a birds-eye view of the entire document's knowledge and detailed sub-maps for specific concepts.

## 🛠️ Technology Stack
- **Python**: Core logic.
- **Streamlit**: Modern and interactive web interface.
- **SpaCy**: Advanced NLP for relationship extraction and lemmatization.
- **NetworkX**: Graph theory and network analysis.
- **Matplotlib**: Static graph generation.
- **PyPDF2**: PDF text extraction.

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd ConceptMapGenerator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🧠 How it Works
The engine uses a dual-layered extraction approach:
1. **Linguistic Patterns**: It looks for specific markers like *"X such as Y"*, *"X including Y"*, and *"X is a type of Y"*.
2. **Dependency Trees**: It analyzes the syntactic structure of sentences to find subjects and objects linked by specific verbs (e.g., *consist of*, *contain*, *include*).

---
*Developed as part of the Text Mining course portfolio.*
