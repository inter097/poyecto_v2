import re 
import spacy
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os
import PyPDF2

# Configuración de página
st.set_page_config(page_title="Concept Map Generator", page_icon="🧠", layout="wide")

# Cargar modelo de SpaCy
@st.cache_resource
def load_nlp():
    model_path = os.path.join(os.getcwd(), "spacy_model", "en_core_web_sm")
    if os.path.exists(model_path):
        return spacy.load(model_path)
    else:
        # Fallback para Streamlit Cloud
        try:
            return spacy.load("en_core_web_sm")
        except:
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")

nlp = load_nlp()

# Palabras de parada personalizadas
custom_stopwords = {"and", "or", "but", "the", "a", "an", "are", "is", "was", "were", "be", "being", "been"}

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Función para limpiar hipónimos
def clean_hyponyms(hyponyms_text):
    doc = nlp(hyponyms_text)
    clean = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            if token.text.lower() not in custom_stopwords:
                clean.append(token.lemma_.lower())
    return list(set(clean))

# Expresiones regulares para patrones
patterns = [
    r"(?P<hypernym>\w+)\s+is\s+a\s+hypernym\s+of\s+(?P<hyponyms>[\w\s,]+)",  # "X is a hypernym of Y"
    r"(?P<hypernym>\w+)\s+such\s+as\s+(?P<hyponyms>[\w\s,]+)",  # "X such as Y"
    r"(?P<hyponyms>[\w\s,]+)\s+is\s+a\stype\sof\s+(?P<hypernym>\w+)",  # "Y is a type of X"
    r"(?P<hypernym>\w+)\s+including\s+(?P<hyponyms>[\w\s,]+)",  # "X including Y"
    r"(?P<hyponyms>[\w\s,]+)\s+is\s+a\skind\sof\s+(?P<hypernym>\w+)",  # "Y is a kind of X"
    r"(?P<hypernym>\w+)\s+consists\sof\s+(?P<hyponyms>[\w\s,]+)",  # "X consists of Y"
    r"(?P<hypernym>\w+)\sand\s+other\s+(?P<hyponyms>\w+)",  # "X and other Y"
    r"(?P<hypernym>\w+)\s+or\s+similar\s+(?P<hyponyms>\w+)",  # "X or similar Y"
    r"(?P<hyponyms>\w+)\s+belong\sto\sthe\scategory\sof\s+(?P<hypernym>\w+)",  # "Y belong to the category of X"
    r"(?P<hyponyms>\w+)\s+categorized\sas\s+(?P<hypernym>\w+)",  # "Y categorized as X"
    r"examples\sof\s(?P<hypernym>\w+)\s+include\s+(?P<hyponyms>[\w\s,]+)",  # "Examples of X include Y"
    r"(?P<hypernym>\w+)\s+divided\sinto\s+(?P<hyponyms>[\w\s,]+)",  # "X divided into Y"
    r"(?P<hypernym>\w+)\s+characterized\sby\s+(?P<hyponyms>[\w\s,]+)",  # "X characterized by Y"
    r"(?P<hypernym>\w+)\s+comprising\s+(?P<hyponyms>[\w\s,]+)",  # "X comprising Y"
    r"(?P<hypernym>\w+)\s+like\s+(?P<hyponyms>[\w\s,]+)"  # "X like Y"
]

# Función para extraer relaciones usando regex y dependencias
def extract_relationships(text):
    pairs = []
    
    # 1. Regex (Hearst Patterns)
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            hyper = match.group("hypernym").strip().lower()
            hypos_raw = match.group("hyponyms")
            hypos_clean = clean_hyponyms(hypos_raw)
            for hypo in hypos_clean:
                pairs.append((hyper, hypo))
                
    # 2. Dependency Parsing (SVO and attributes)
    doc = nlp(text)
    for sent in doc.sents:
        for token in sent:
            # Look for subjects (nouns) connected to objects via "is", "contains", "includes"
            if token.pos_ == "VERB" and token.lemma_ in ["be", "include", "contain", "comprise", "consist"]:
                subj = [w.lemma_.lower() for w in token.lefts if w.dep_ == "nsubj" and w.pos_ in ["NOUN", "PROPN"]]
                objs = [w.lemma_.lower() for w in token.rights if w.dep_ in ["attr", "dobj", "pobj"] and w.pos_ in ["NOUN", "PROPN"]]
                for s in subj:
                    for o in objs:
                        if s not in custom_stopwords and o not in custom_stopwords:
                            pairs.append((s, o))
                            
    return list(set(pairs))

# Función para generar el código Mermaid
def generate_mermaid_code(grouped_pairs):
    mermaid_code = "graph TD\n"
    mermaid_code += "  Conceptual_Map\n"

    for hypernym, hyponyms in grouped_pairs.items():
        # Limpiar nombres para Mermaid
        hyper_clean = re.sub(r'\W+', '_', hypernym)
        mermaid_code += f"  Conceptual_Map --> {hyper_clean}[{hypernym}]\n"
        for hypo in hyponyms:
            hypo_clean = re.sub(r'\W+', '_', hypo)
            mermaid_code += f"  {hyper_clean} --> {hypo_clean}[{hypo}]\n"

    return mermaid_code

# Función para construir y guardar los mapas conceptuales
def build_and_save_concept_maps(grouped_pairs):
    root_node = "Conceptual Map"
    global_graph = nx.DiGraph()
    saved_images = []

    for hypernym, hyponyms in grouped_pairs.items():
        G = nx.DiGraph()
        for hypo in hyponyms:
            G.add_edge(hypernym, hypo)
            global_graph.add_edge(root_node, hypernym)
            global_graph.add_edge(hypernym, hypo)

        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="#ADF7D1", font_size=12, font_weight="bold", arrows=True, width=2)
        plt.title(f"Concept Map: {hypernym}", fontsize=16)
        filename = f"concept_map_{re.sub(r'\W+', '_', hypernym)}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        saved_images.append(filename)

    if not global_graph.nodes:
        return []

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(global_graph, k=0.3, seed=42)
    nx.draw(global_graph, pos, with_labels=True, node_size=3500, node_color="#AEC6CF", font_size=10, font_weight="bold", arrows=True, width=1.5)
    plt.title("Global Concept Map of the Document", fontsize=20)
    global_filename = "concept_map_global.png"
    plt.savefig(global_filename, bbox_inches='tight')
    plt.close()
    saved_images.append(global_filename)

    return saved_images

# Interfaz de usuario con Streamlit
def main():
    st.title("🧠 Automated Concept Map Generator")
    st.markdown("### Extract knowledge and visualize relationships from your text or PDF files.")

    with st.sidebar:
        st.header("📂 Upload or Paste Content")
        upload_choice = st.radio("Choose source:", ("Text Area", "Upload PDF", "Upload TXT"))
        
        input_text = ""
        if upload_choice == "Text Area":
            input_text = st.text_area("✍️ Paste your text here:", height=300)
        elif upload_choice == "Upload PDF":
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file is not None:
                input_text = extract_text_from_pdf(uploaded_file)
        elif upload_choice == "Upload TXT":
            uploaded_file = st.file_uploader("Choose a TXT file", type="txt")
            if uploaded_file is not None:
                input_text = str(uploaded_file.read(), "utf-8")

    if st.button("📌 Generate Concept Map"):
        if not input_text.strip():
            st.warning("Please provide some text to analyze.")
            return

        with st.spinner("Analyzing text and building relationships..."):
            pairs = extract_relationships(input_text)

            if not pairs:
                st.info("No conceptual relationships were found in the text.")
            else:
                grouped = defaultdict(list)
                for hyper, hypo in pairs:
                    grouped[hyper].append(hypo)

                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Visual Map", "📋 Extracted Relations", "📜 Code (Mermaid)"])

                with tab1:
                    saved_files = build_and_save_concept_maps(grouped)
                    if saved_files:
                        # Show global map first if exists
                        global_map = next((f for f in saved_files if "global" in f), None)
                        if global_map:
                            st.image(Image.open(global_map), caption="Full Document Knowledge Map", use_container_width=True)
                        
                        st.divider()
                        st.subheader("Sub-maps by Concept")
                        cols = st.columns(2)
                        for i, filename in enumerate(f for f in saved_files if "global" not in f):
                            with cols[i % 2]:
                                if os.path.exists(filename):
                                    st.image(Image.open(filename), caption=f"Local map for concept", use_container_width=True)

                with tab2:
                    st.subheader(f"Found {len(grouped)} key concepts")
                    for hyper, hypos in grouped.items():
                        st.markdown(f"**{hyper.capitalize()}** ➔ {', '.join(hypos)}")

                with tab3:
                    mermaid_code = generate_mermaid_code(grouped)
                    st.code(mermaid_code, language="mermaid")
                    st.markdown(f"[View in Mermaid Live Editor](https://mermaid.live/edit#base64={os.popen(f'echo {mermaid_code} | base64').read().strip()})")

if __name__ == "__main__":
    main()
