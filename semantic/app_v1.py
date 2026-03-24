import spacy
import nltk
from nltk.corpus import wordnet as wn
import streamlit as st
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# Descargar recursos de NLTK
nltk.download('wordnet')
nltk.download('omw-1.4')

# Cargar modelo de SpaCy
nlp = spacy.load("en_core_web_sm")

# Función para obtener los synsets de una palabra y sus relaciones (solo hiperonimia)
def get_hypernyms(word):
    synsets = wn.synsets(word)
    hypernyms = []
    for syn in synsets:
        for hyper in syn.hypernyms():
            # Asegurarse de que solo tomamos hiperonimia de palabras de tipo sustantivo
            if hyper.pos() == 'n':  # Solo sustantivos
                hypernyms.append(hyper.name().split('.')[0])  # Obtener el nombre del hiperónimo
    return hypernyms

# Función para extraer sustantivos y sus relaciones de hiperonimia/hiponimia usando WordNet
def extract_hypernyms_from_text(text):
    doc = nlp(text)
    relations = []
    
    # Extraer sustantivos
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:  # Solo sustantivos y nombres propios
            word = token.text.lower()
            hypernyms = get_hypernyms(word)
            if hypernyms:
                for hyper in hypernyms:
                    # Filtrar relaciones que no sean jerárquicas o que no sean sustantivos
                    if word != hyper:  # Evitar incluir la misma palabra como hiperónimo
                        relations.append((hyper, word))  # Relación de hiperonimia: (hiperónimo, hipónimo)
    
    return relations

# Función para generar el código Mermaid
def generate_mermaid_code(grouped_pairs):
    mermaid_code = "graph TD\n"
    mermaid_code += "  Conceptual_Map\n"  # Nodo superior llamado Conceptual Map

    # Agregar relaciones entre Conceptual Map y cada hiperónimo
    for hypernym, hyponyms in grouped_pairs.items():
        mermaid_code += f"  Conceptual_Map --> {hypernym}\n"  # Conectar Conceptual Map con cada hiperónimo
        for hypo in hyponyms:
            mermaid_code += f"  {hypernym} --> {hypo}\n"  # Relación de hiperonimia --> hipónimo

    return mermaid_code

# Función para construir y guardar los mapas conceptuales
def build_and_save_concept_maps(grouped_pairs):
    root_node = "Conceptual Map"
    global_graph = nx.DiGraph()

    saved_images = []

    # Crear cada relación individualmente y agregar hipónimos a la relación global
    for hypernym, hyponyms in grouped_pairs.items():
        G = nx.DiGraph()
        for hypo in hyponyms:
            G.add_edge(hypernym, hypo)
            # Asegurarnos de agregar cada hipónimo al gráfico global
            global_graph.add_edge(root_node, hypernym)  # Conectar Conceptual Map con el hiperónimo
            global_graph.add_edge(hypernym, hypo)  # Conectar hiperónimo con hipónimo

        # Guardar la imagen del mapa conceptual individual
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
        plt.title(f"Concept Map: {hypernym}", fontsize=14)
        filename = f"concept_map_{hypernym}.png"
        plt.savefig(filename)
        plt.close()
        saved_images.append(filename)

    # Generar el gráfico global con todas las relaciones
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
    st.title("🧠 Concept Map Generator (Basic Hypernyms Extraction using WordNet)")

    input_text = st.text_area("✍️ Paste or type your text here:", height=200)

    if st.button("📌 Generate Concept Maps"):
        if not input_text.strip():
            st.warning("Please enter a text to analyze.")
            return

        # Mostrar texto después de preprocesamiento
        st.subheader("🧹 Processed Text:")
        st.write(input_text)  # Muestra el texto original

        relations = extract_hypernyms_from_text(input_text)

        if not relations:
            st.info("No hypernym/hyponym relationships found.")
        else:
            # Agrupar relaciones por hiperónimo
            grouped = defaultdict(list)
            for hyper, hypo in relations:
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