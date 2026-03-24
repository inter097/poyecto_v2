import re
import spacy
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import os

# Cargar modelo de SpaCy
model_path = os.path.join(os.getcwd(), "spacy_model", "en_core_web_sm")
nlp = spacy.load(model_path)

# Palabras de parada personalizadas
custom_stopwords = {"and", "or", "but", "the", "a", "an", "are", "is", "was", "were", "be", "being", "been"}

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
    r"(?P<hypernym>\w+)\s+is\s+a\s+hypernym\s+of\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hypernym>\w+)\s+such\s+as\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hyponyms>[\w\s,]+)\s+is\s+a\s+type\s+of\s+(?P<hypernym>\w+)",
    r"(?P<hypernym>\w+)\s+including\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hyponyms>[\w\s,]+)\s+is\s+a\s+kind\s+of\s+(?P<hypernym>\w+)",
    r"(?P<hypernym>\w+)\s+consists\s+of\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hypernym>\w+)\s+and\s+other\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hypernym>\w+)\s+or\s+similar\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hyponyms>\w+)\s+belong\s+to\s+the\s+category\s+of\s+(?P<hypernym>\w+)",
    r"(?P<hyponyms>\w+)\s+categorized\s+as\s+(?P<hypernym>\w+)",
    r"examples\s+of\s+(?P<hypernym>\w+)\s+include\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hypernym>\w+)\s+divided\s+into\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hypernym>\w+)\s+characterized\s+by\s+(?P<hyponyms>[\w\s,]+)",
    r"(?P<hypernym>\w+)\s+comprising\s+(?P<hyponyms>[\w\s,]+)"
]

# Función para extraer relaciones de hiperonimia e hiponimia
def extract_hyponym_patterns(text):
    pairs = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            hyper = match.group("hypernym").strip().lower()
            hypos_raw = match.group("hyponyms")
            hypos_clean = clean_hyponyms(hypos_raw)
            for hypo in hypos_clean:
                pairs.append((hyper, hypo))
    return pairs

# Función para generar el código Mermaid
def generate_mermaid_code(grouped_pairs):
    mermaid_code = "graph TD\n"  # Inicio del diagrama Mermaid
    for hypernym, hyponyms in grouped_pairs.items():
        for hypo in hyponyms:
            mermaid_code += f"  {hypernym} --> {hypo}\n"  # Formato de relación: Hiperónimo --> Hipónimo
    return mermaid_code

# Función para construir y guardar los mapas conceptuales
def build_and_save_concept_maps(grouped_pairs):
    root_node = "Concept Map of the Document"
    global_graph = nx.DiGraph()

    saved_images = []

    for hypernym, hyponyms in grouped_pairs.items():
        G = nx.DiGraph()
        for hypo in hyponyms:
            G.add_edge(hypernym, hypo)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
        plt.title(f"Concept Map: {hypernym}", fontsize=14)
        filename = f"concept_map_{hypernym}.png"
        plt.savefig(filename)
        plt.close()
        saved_images.append(filename)

        global_graph.add_edge(root_node, hypernym)
        for hypo in hyponyms:
            global_graph.add_edge(hypernym, hypo)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(global_graph, seed=42)
    nx.draw(global_graph, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
    plt.title("Global Concept Map of the Document", fontsize=16)
    global_filename = "concept_map_global.png"
    plt.savefig(global_filename)
    plt.close()
    saved_images.append(global_filename)

    return saved_images

# Interfaz de usuario con Streamlit
def main():
    st.title("🧠 Concept Map Generator (Text Mining)")

    input_text = st.text_area("✍️ Paste or type your text here:", height=200)

    if st.button("📌 Generate Concept Maps"):
        if not input_text.strip():
            st.warning("Please enter a text to analyze.")
            return

        pairs = extract_hyponym_patterns(input_text)

        if not pairs:
            st.info("No hypernym/hyponym relationships found.")
        else:
            grouped = defaultdict(list)
            for hyper, hypo in pairs:
                grouped[hyper].append(hypo)

            st.subheader(f"🔍 Relationships found: {len(grouped)}")
            for hyper, hypos in grouped.items():
                st.markdown(f"**{hyper.capitalize()}** → {', '.join(hypos)}")

            st.subheader("🧭 Concept Maps Generated:")
            saved_files = build_and_save_concept_maps(grouped)
            for filename in saved_files:
                if os.path.exists(filename):
                    st.image(Image.open(filename), caption=filename, use_container_width=True)

            # Generate Mermaid code
            mermaid_code = generate_mermaid_code(grouped)
            st.subheader("📜 Mermaid Code of the Concept Map:")
            st.code(mermaid_code, language="mermaid")

            # Add link to Mermaid viewer
            mermaid_viewer_url = "https://mermaid-js.github.io/mermaid-live-editor/"
            st.markdown(f"[View the concept map in the Mermaid viewer]( {mermaid_viewer_url} )", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
