import re
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
from PIL import Image
import os
import nltk
from nltk.corpus import wordnet as wn

# Descargar recursos necesarios para WordNet
nltk.download('wordnet')

# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")  # O usar el modelo local si es necesario

# Stopwords personalizadas
custom_stopwords = {"and", "or", "but", "the", "a", "an", "are", "is", "was", "were", "be", "being", "been"}

# Función para limpiar hipónimos (Lematización y minúsculas)
def clean_hyponyms(hyponyms_text):
    doc = nlp(hyponyms_text)
    clean = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:  # Solo sustantivos y nombres propios
            if token.text.lower() not in custom_stopwords:
                clean.append(token.lemma_.lower())  # Usamos la forma lematizada
    return list(set(clean))

# Función para validar si hay relación hiperónimo/hipónimo usando WordNet
def is_hypernym(hyper, hypo):
    synsets_hyper = wn.synsets(hyper)
    synsets_hypo = wn.synsets(hypo)
    for hypo_syn in synsets_hypo:
        for hyper_syn in synsets_hyper:
            if hyper_syn in hypo_syn.hypernyms():  # Verifica si hay relación de hiperónimo
                return True
    return False

# Ampliar los patrones de Hearst
def extract_hyponym_patterns(text):
    patterns = [
        r"(?P<hypernym>\w+)\s+(?:such as|including|especially|like)\s+(?P<hyponyms>[\w\s,]+)",
        r"(?P<hyponyms>[\w\s,]+)\s+(?:are types of|are kinds of)\s+(?P<hypernym>\w+)",
        r"(?P<hypernym>\w+)\s+is a hypernym of\s+(?P<hyponyms>[\w\s,]+)",
        r"(?P<hyponyms>[\w\s,]+)\s+are hyponyms of the word\s+[\"']?(?P<hypernym>\w+)[\"']?",
        r"(?P<hypernym>\w+)\s+includes\s+(?P<hyponyms>[\w\s,]+)",
        r"(?P<hyponyms>[\w\s,]+)\s+are examples of\s+(?P<hypernym>\w+)",
        r"(?P<hypernym>\w+)\s+such as\s+(?P<hyponyms>[\w\s,]+)",
        r"(?P<hyponyms>[\w\s,]+)\s+are members of the class\s+(?P<hypernym>\w+)",
        r"(?P<hyponyms>[\w\s,]+)\s+are kinds of\s+(?P<hypernym>\w+)"
    ]
    
    pairs = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            hyper = match.group("hypernym").strip().lower()
            hypos_raw = match.group("hyponyms")
            hypos_clean = clean_hyponyms(hypos_raw)
            for hypo in hypos_clean:
                if is_hypernym(hyper, hypo):  # Validar si la relación es real usando WordNet
                    pairs.append((hyper, hypo))

    grouped = defaultdict(list)
    for hyper, hypo in pairs:
        grouped[hyper].append(hypo)

    return pairs, grouped

# Construir mapas conceptuales y grafo global
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

# Función principal para generar mapas conceptuales
def generate_concept_maps(text):
    st.title("🧠 Generador de Mapas Conceptuales (Text Mining)")
    st.write("Este sistema detecta relaciones de hiperonimia e hiponimia y construye mapas conceptuales a partir del texto proporcionado.")

    input_text = st.text_area("✍️ Escribe o pega tu texto aquí:", height=200)

    if st.button("📌 Generar Mapas Conceptuales"):
        if not input_text.strip():
            st.warning("Por favor, escribe un texto para analizar.")
            return

        pairs, grouped = extract_hyponym_patterns(input_text)

        if not pairs:
            st.info("No se encontraron relaciones de hiperonimia/hiponimia.")
        else:
            st.subheader(f"🔍 Relaciones encontradas: {len(grouped)}")
            for hyper, hypos in grouped.items():
                st.markdown(f"**{hyper.capitalize()}** → {', '.join(hypos)}")

            st.subheader("🧭 Mapas Conceptuales Generados:")
            saved_files = build_and_save_concept_maps(grouped)
            for filename in saved_files:
                if os.path.exists(filename):
                    st.image(Image.open(filename), caption=filename, use_container_width=True)

# Ejecutar ejemplo
if __name__ == "__main__":
    generate_concept_maps('')
