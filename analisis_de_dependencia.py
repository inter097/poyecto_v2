import spacy
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import os
import re

# Cargar modelo de SpaCy para análisis de dependencias en inglés
nlp = spacy.load("en_core_web_sm")

# Función para preprocesar el texto (eliminar puntuación y números)
def preprocesar_texto(texto):
    # Eliminar puntuación y números, dejar solo letras y espacios
    texto_limpio = re.sub(r'[^a-zA-Z\s]', '', texto)
    return texto_limpio.lower()

# Función para extraer las relaciones de dependencias entre palabras
def extraer_dependencias(texto):
    # Preprocesar el texto antes de extraer
    texto_limpio = preprocesar_texto(texto)
    doc = nlp(texto_limpio)
    relaciones = []

    for token in doc:
        # Verificar si el token tiene una relación de dependencia válida
        if token.dep_ in ['nsubj', 'dobj', 'prep', 'pobj']:
            # Buscamos relaciones de hiperonimia/hiponimia
            # Ejemplo: 'is a type of', 'is a kind of'
            if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                # Si la relación es un verbo que conecta un sustantivo con otro sustantivo
                hiperonimo = token.head.text.lower()
                hiponimo = token.text.lower()
                relaciones.append((hiperonimo, hiponimo))
    
    return relaciones

# Función para generar el código Mermaid
def generar_codigo_mermaid(relaciones_agrupadas):
    codigo_mermaid = "graph TD\n"
    codigo_mermaid += "  Mapa_Conceptual\n"

    for hiperonimo, hiponimos in relaciones_agrupadas.items():
        codigo_mermaid += f"  Mapa_Conceptual --> {hiperonimo}\n"
        for hip in hiponimos:
            codigo_mermaid += f"  {hiperonimo} --> {hip}\n"

    return codigo_mermaid

# Función para construir y guardar los mapas conceptuales
def construir_guardar_mapas_conceptuales(relaciones_agrupadas):
    nodo_raiz = "Mapa Conceptual"
    grafo_global = nx.DiGraph()
    imagenes_guardadas = []

    for hiperonimo, hiponimos in relaciones_agrupadas.items():
        G = nx.DiGraph()
        for hip in hiponimos:
            G.add_edge(hiperonimo, hip)
            grafo_global.add_edge(nodo_raiz, hiperonimo)
            grafo_global.add_edge(hiperonimo, hip)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
        plt.title(f"Mapa Conceptual: {hiperonimo}", fontsize=14)
        filename = f"mapa_conceptual_{hiperonimo}.png"
        plt.savefig(filename)
        plt.close()
        imagenes_guardadas.append(filename)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(grafo_global, seed=42)
    nx.draw(grafo_global, pos, with_labels=True, node_size=2000, font_size=10, arrows=True)
    plt.title("Mapa Conceptual Global del Documento", fontsize=16)
    filename_global = "mapa_conceptual_global.png"
    plt.savefig(filename_global)
    plt.close()
    imagenes_guardadas.append(filename_global)

    return imagenes_guardadas

# Interfaz de usuario con Streamlit
def main():
    st.title("Generador de Mapas Conceptuales (Detección de Hiperonimia e Hiponimia)")

    texto_entrada = st.text_area("Ingresa tu texto aquí:", height=200)

    if st.button("Generar Mapas Conceptuales"):
        if not texto_entrada.strip():
            st.warning("Por favor, ingresa un texto para analizar.")
            return

        st.subheader("Texto Procesado:")
        st.write(texto_entrada)

        relaciones = extraer_dependencias(texto_entrada)

        if not relaciones:
            st.info("No se encontraron relaciones de hiperonimia/hiponimia.")
        else:
            relaciones_agrupadas = defaultdict(list)
            for hip, hipon in relaciones:
                relaciones_agrupadas[hip].append(hipon)

            st.subheader(f"Relaciones encontradas: {len(relaciones_agrupadas)}")
            for hip, hiponimos in relaciones_agrupadas.items():
                st.markdown(f"**{hip.capitalize()}** → {', '.join(hiponimos)}")

            st.subheader("Mapas Conceptuales Generados:")
            imagenes = construir_guardar_mapas_conceptuales(relaciones_agrupadas)
            for imagen in imagenes:
                if os.path.exists(imagen):
                    st.image(Image.open(imagen), caption=imagen, use_container_width=True)

            # Generar código Mermaid
            codigo_mermaid = generar_codigo_mermaid(relaciones_agrupadas)
            st.subheader("Código Mermaid del Mapa Conceptual:")
            st.code(codigo_mermaid, language="mermaid")

            # Enlace al visualizador de Mermaid
            url_visualizador_mermaid = "https://mermaid-js.github.io/mermaid-live-editor/"
            st.markdown(f"[Ver el mapa conceptual en el visualizador de Mermaid]( {url_visualizador_mermaid} )", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
