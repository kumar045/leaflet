import streamlit as st
import PyPDF2
import google.generativeai as genai
import textstat
import logging
import math
import re
import numpy as np
import pandas as pd
import os
import sys
import requests
import tarfile
import spacy
from spacy.lang.de import German

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_spacy_model():
    nlp = German()
    nlp.add_pipe('sentencizer')
    return nlp

nlp = load_spacy_model()

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        return ""

def initialize_gemini_client(api_key):
    """Initialize the Google Gemini client with the provided API key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name="gemini-1.5-pro-exp-0801")

def simplify_text_with_gemini(text, api_key, metrics=None):
    """Use Google Gemini to simplify the given German text, optionally using current metrics."""
    try:
        if not text.strip():
            return "Error: No text provided for simplification."

        model = initialize_gemini_client(api_key)
        chat_session = model.start_chat(history=[])

        prompt = f"""
        Sie sind ein KI-Assistent, der sich auf die Vereinfachung pharmazeutischer und medizinischer Anweisungen spezialisiert hat. Ihre Aufgabe ist es, den gegebenen Text so umzuschreiben, dass er für Menschen mit begrenzter Gesundheitskompetenz leicht verständlich ist, mit einem Leseniveau eines 14-Jährigen als Ziel. Befolgen Sie diese Richtlinien:

        ##Aufgabe:##
        Verfassen Sie Anweisungen zu pharmazeutischen und medizinischen Themen so um, dass sie für Personen mit eingeschränkter Gesundheitskompetenz und einem Leselevel auf dem Niveau eines 14-Jährigen einfach verständlich sind. Achten Sie dabei besonders darauf, dass alle rechtlichen und sicherheitsrelevanten Informationen klar und vollständig präsentiert werden. Dies beinhaltet spezifische Hinweise für spezielle Bevölkerungsgruppen wie Schwangere oder Stillende sowie klare Anweisungen zu Maßnahmen bei einer Überdosierung.

        ##Spezifika:##
        Reduzieren Sie so weit wie möglich:
        - lange Sätze: maximal 20 Wörter pro Satz
        - Schachtelsätze: Keine verschachtelten Sätze, Aufspaltung in mehrere Sätze
        - Nebensätze: maximal zwei Nebensätze
        - Nominalisierungen: paraphrasieren Sie diese
        - Passivkonstruktionen: Aktivsätze sind Passivsätzen vorzuziehen
        - Partizipialkonstruktionen
        - Negationen
        - lange Wörter
        - mehrsilbige Wörter
        - Komposita: verzichten Sie darauf, sofern das nicht die Sinnhaftigkeit beeinträchtigt
        - Durchkopplungen: nicht mehr als vier Glieder
        - Fachbegriffe: Übersetzen Sie medizinische Begriffe in eine patientenverständliche Sprache, Fachausdrücke sind nach einer umgangssprachlichen Entsprechung in Klammern zu platzieren

        Beachten Sie:
        - Direkte Ansprache: Verwenden Sie die persönliche Anrede mit Imperativen. Die Anrede sollte kurz, konkret und anschaulich sein.
        - Semantische Zusammenhänge: Bauen Sie eine semantische Relation auf, indem Sie zunächst die Ursache und dann die Wirkung nennen.
        - Quantifizierung: Quantifizieren Sie Angaben so, dass der Anwender diese direkt versteht.

        Berücksichtigen Sie unbedingt Hinweise zur Anwendung während der Schwangerschaft und Stillzeit sowie zu Maßnahmen bei Überdosierung.

        Bitte vereinfachen Sie den folgenden deutschen Text:

        {text}

        Stellen Sie sicher, dass Ihre vereinfachte Version alle wichtigen Informationen beibehält und dabei für Leser mit begrenzter Gesundheitskompetenz besser zugänglich ist. Verwenden Sie durchgehend die formelle "Sie"-Form.
        """

        if metrics:
            prompt += f"\n\nAktuelle Lesbarkeitsmetriken: {metrics}\nBitte verbessern Sie diese Metriken in Ihrer Vereinfachung."

        response = chat_session.send_message(prompt)

        if not response.text.strip():
            return "Error: The model returned an empty response. Please try again with a different text or check your API key."

        return response.text
    except Exception as e:
        logger.error(f"Fehler in simplify_text_with_gemini: {str(e)}")
        return f"Fehler: Anfrage konnte nicht verarbeitet werden. Bitte versuchen Sie es später erneut. Details: {str(e)}"

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        if not text.strip():
            continue  # Skip empty texts
        try:
            # Chunk the text if it's too long
            chunks = [text[i:i+5000] for i in range(0, len(text), 5000)]
            chunk_embeddings = []
            for chunk in chunks:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk,
                    task_type="retrieval_document"
                )
                chunk_embeddings.append(result['embedding'])
            # Average the embeddings if there were multiple chunks
            if chunk_embeddings:
                embeddings.append(np.mean(chunk_embeddings, axis=0))
        except Exception as e:
            logger.error(f"Error getting embedding for text: {str(e)}")
    return embeddings

def two_phase_approach(original_text, simplified_text):
    """Implement a two-phase approach inspired by the KnowHalu framework for German text."""
    original_key_phrases = set(re.findall(r'\b\w+(?:\s+\w+){2,3}\b', original_text))
    simplified_key_phrases = set(re.findall(r'\b\w+(?:\s+\w+){2,3}\b', simplified_text))
    
    non_fabrication_score = len(original_key_phrases.intersection(simplified_key_phrases)) / len(original_key_phrases) if original_key_phrases else 1.0

    original_entities = set(ent.text for ent in nlp(original_text).ents)
    simplified_entities = set(ent.text for ent in nlp(simplified_text).ents)
    
    factual_accuracy_score = len(original_entities.intersection(simplified_entities)) / len(original_entities) if original_entities else 1.0

    return {
        'non_fabrication_score': non_fabrication_score,
        'factual_accuracy_score': factual_accuracy_score
    }
    
def cosine_similarity(embeddings1, embeddings2):
    similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            similarity_matrix[i, j] = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity_matrix

def coverage_accuracy_assessment(original_text, simplified_text):
    """Assess coverage and accuracy of simplified text compared to original."""
    try:
        original_sentences = [sent.text.strip() for sent in nlp(original_text).sents if sent.text.strip()]
        simplified_sentences = [sent.text.strip() for sent in nlp(simplified_text).sents if sent.text.strip()]

        if not original_sentences or not simplified_sentences:
            return {
                'coverage_score': 0,
                'covered_sentences': 0,
                'total_original_sentences': len([sent for sent in nlp(original_text).sents]),
                'warning': "No non-empty sentences found in original or simplified text."
            }

        original_embeddings = get_embeddings(original_sentences)
        simplified_embeddings = get_embeddings(simplified_sentences)

        if not original_embeddings or not simplified_embeddings:
            return {
                'coverage_score': 0,
                'covered_sentences': 0,
                'total_original_sentences': len(original_sentences),
                'warning': "Failed to generate embeddings for sentences."
            }

        similarity_matrix = cosine_similarity(original_embeddings, simplified_embeddings)

        covered_sentences = sum(similarity_matrix.max(axis=1) > 0.8)  # Threshold can be adjusted
        coverage_score = covered_sentences / len(original_sentences) if original_sentences else 0

        return {
            'coverage_score': coverage_score,
            'covered_sentences': covered_sentences,
            'total_original_sentences': len(original_sentences)
        }
    except Exception as e:
        logger.error(f"Error in coverage_accuracy_assessment: {str(e)}")
        return {
            'coverage_score': 0,
            'covered_sentences': 0,
            'total_original_sentences': 0,
            'error': str(e)
        }
        
def verify_medical_entities(original_text, simplified_text):
    """Verify that medical entities in the original German text are preserved in the simplified text."""
    original_doc = nlp(original_text)
    simplified_doc = nlp(simplified_text)

    original_entities = set((ent.text.lower(), ent.label_) for ent in original_doc.ents if ent.label_ in ['DRUG', 'DISEASE', 'SYMPTOM'])
    simplified_entities = set((ent.text.lower(), ent.label_) for ent in simplified_doc.ents if ent.label_ in ['DRUG', 'DISEASE', 'SYMPTOM'])

    preserved_entities = original_entities.intersection(simplified_entities)
    missing_entities = original_entities - simplified_entities

    preservation_score = len(preserved_entities) / len(original_entities) if original_entities else 1.0

    return {
        'preservation_score': preservation_score,
        'preserved_entities': list(preserved_entities),
        'missing_entities': list(missing_entities)
    }

def self_consistency_check(original_text, simplified_text, api_key, num_versions=2):
    """Generate multiple simplified versions of German text and compare them for consistency."""
    versions = [simplify_text_with_gemini(original_text, api_key) for _ in range(num_versions)]
    versions.append(simplified_text)

    embeddings = get_embeddings(versions)
    similarities = cosine_similarity(embeddings, embeddings)

    consistency_score = similarities.mean()

    return {
        'consistency_score': consistency_score,
        'versions': versions
    }

def citation_accuracy_check(original_text, simplified_text):
    """Check if citations in the original German text are preserved in the simplified text."""
    original_citations = re.findall(r'\[(\d+)\]', original_text)
    simplified_citations = re.findall(r'\[(\d+)\]', simplified_text)

    preserved_citations = set(original_citations).intersection(set(simplified_citations))
    citation_accuracy = len(preserved_citations) / len(original_citations) if original_citations else 1.0

    return {
        'citation_accuracy': citation_accuracy,
        'original_citations': original_citations,
        'preserved_citations': list(preserved_citations)
    }

def implement_safeguards(simplified_text):
    """Implement safeguards and guardrails to ensure critical information is preserved in German text."""
    safeguards = [
        ("dosage", r"\b\d+\s*(mg|g|ml)\b"),
        ("warning", r"\b(Warnung|Vorsicht|Achtung)\b"),
        ("side[- ]effects?", r"\b(Nebenwirkungen?|Nebenwirkung)\b"),
    ]

    results = {}
    for name, pattern in safeguards:
        if re.search(pattern, simplified_text, re.IGNORECASE):
            results[name] = "Vorhanden"
        else:
            results[name] = "Fehlend"

    return results

def factuality_faithfulness_check(original_text, simplified_text):
    """Check factuality and faithfulness of the simplified German text."""
    try:
        original_doc = nlp(original_text)
        simplified_doc = nlp(simplified_text)

        original_entities = set(ent.text for ent in original_doc.ents)
        simplified_entities = set(ent.text for ent in simplified_doc.ents)

        if len(original_entities) == 0:
            factuality_score = 1.0  # If no entities in original, assume perfect factuality
        else:
            factuality_score = len(original_entities.intersection(simplified_entities)) / len(original_entities)

        original_sentences = [sent.text for sent in original_doc.sents]
        simplified_sentences = [sent.text for sent in simplified_doc.sents]
        
        if not original_sentences or not simplified_sentences:
            return {
                'factuality_score': factuality_score,
                'faithfulness_score': 1.0,
                'warning': "No sentences found in original or simplified text."
            }

        original_embeddings = get_embeddings(original_sentences)
        simplified_embeddings = get_embeddings(simplified_sentences)
        
        if not original_embeddings or not simplified_embeddings:
            return {
                'factuality_score': factuality_score,
                'faithfulness_score': 0.0,
                'warning': "Failed to generate embeddings for sentences."
            }

        similarities = cosine_similarity(original_embeddings, simplified_embeddings)
        faithfulness_score = similarities.max(axis=1).mean()

        return {
            'factuality_score': factuality_score,
            'faithfulness_score': faithfulness_score
        }
    except Exception as e:
        logger.error(f"Error in factuality_faithfulness_check: {str(e)}")
        return {
            'factuality_score': 0,
            'faithfulness_score': 0,
            'error': str(e)
        }

def count_syllables(word):
    """Count the number of syllables in a German word."""
    word = word.lower()
    count = 0
    vowels = "aeiouäöüy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith("le"):
        count += 1
    if count == 0:
        count += 1
    return count

def calculate_german_metrics(text):
    """Calculate German-specific readability metrics."""
    words = re.findall(r'\w+', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(count_syllables(word) for word in words)
    
    # Calculate metrics
    avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
    avg_syllables_per_word = total_syllables / total_words if total_words > 0 else 0
    
    # Wiener Sachtextformel
    words_with_3plus_syllables = sum(1 for word in words if count_syllables(word) >= 3)
    wiener_sachtextformel = 0.2656 * total_words + 0.2744 * words_with_3plus_syllables - 1.693
    
    # LIX (Läsbarhetsindex)
    long_words = sum(1 for word in words if len(word) > 6)
    lix = (total_words / total_sentences) + (long_words * 100 / total_words) if total_words > 0 and total_sentences > 0 else 0
    
    # German SMOG (G-SMOG)
    g_smog = 7.3 + (0.7 * math.sqrt(words_with_3plus_syllables / total_sentences * 30) + 0.3) if total_sentences > 0 else 0
    
    return {
        'Wiener Sachtextformel': wiener_sachtextformel,
        'LIX (Läsbarhetsindex)': lix,
        'G-SMOG': g_smog,
        'Average words per sentence': avg_words_per_sentence,
        'Average syllables per word': avg_syllables_per_word,
        'Total words': total_words,
        'Total sentences': total_sentences
    }

def constrained_reasoning(original_text, simplified_text, api_key):
    """Perform constrained reasoning to identify potential hallucinations in German text."""
    try:
        model = initialize_gemini_client(api_key)
        chat_session = model.start_chat(history=[])

        prompt = f"""
        Als KI-Assistent für medizinische Texte ist Ihre Aufgabe, potenzielle Halluzinationen in einer vereinfachten Version eines deutschen medizinischen Dokuments zu identifizieren. Vergleichen Sie den Originaltext mit der vereinfachten Version und markieren Sie alle Informationen in der vereinfachten Version, die:

        1. Im Originaltext nicht vorkommen
        2. Dem Originaltext widersprechen
        3. Wichtige Informationen aus dem Originaltext auslassen

        Originaltext:
        {original_text}

        Vereinfachte Version:
        {simplified_text}

        Bitte geben Sie Ihre Analyse in folgendem Format aus:
        1. [Potenzielle Halluzination]: [Erklärung]
        2. [Potenzielle Halluzination]: [Erklärung]
        ...

        Wenn keine Halluzinationen gefunden wurden, geben Sie "Keine Halluzinationen gefunden" aus.
        """

        response = chat_session.send_message(prompt)

        return response.text
    except Exception as e:
        logger.error(f"Fehler in constrained_reasoning: {str(e)}")
        return f"Fehler: Anfrage konnte nicht verarbeitet werden. Bitte versuchen Sie es später erneut. Details: {str(e)}"

def simplify_document(document_content, api_key, max_iterations=1):
    if not document_content.strip():
        return "Error: The extracted text from the PDF is empty. Please check the PDF file.", None, None, None, None, None

    initial_metrics = calculate_german_metrics(document_content)
    simplified_content = document_content
    iterations = []

    for i in range(max_iterations):
        previous_content = simplified_content
        simplified_content = simplify_text_with_gemini(simplified_content, api_key, initial_metrics)
        
        if simplified_content.startswith("Error:") or simplified_content.startswith("Fehler:"):
            return simplified_content, None, iterations, initial_metrics, None, None

        current_metrics = calculate_german_metrics(simplified_content)
        
        iterations.append({
            'iteration': i+1,
            'content': simplified_content,
            'metrics': current_metrics
        })

        if simplified_content == previous_content:
            break

    check_results = factuality_faithfulness_check(document_content, simplified_content)
    if 'error' in check_results:
        logger.error(f"Error in factuality_faithfulness_check: {check_results['error']}")
        check_results = None

    final_metrics = calculate_german_metrics(simplified_content)
    
    try:
        constrained_reasoning_results = constrained_reasoning(document_content, simplified_content, api_key)
    except Exception as e:
        logger.error(f"Error in constrained_reasoning: {str(e)}")
        constrained_reasoning_results = f"Error in constrained reasoning: {str(e)}"
    
    return simplified_content, check_results, iterations, initial_metrics, final_metrics, constrained_reasoning_results
    
def display_metrics_comparison(initial_metrics, final_metrics):
    df = pd.DataFrame({
        'Metric': initial_metrics.keys(),
        'Initial': initial_metrics.values(),
        'Final': final_metrics.values()
    })
    df['Improvement'] = df['Final'] - df['Initial']
    df['Improvement %'] = (df['Improvement'] / df['Initial']) * 100
    return df

def main():
    st.title("German Medical Document Simplification")

    api_key = st.text_input("Enter your Google API Key", type="password")
    
    uploaded_file = st.file_uploader("Choose a German PDF file", type="pdf")
    
    if uploaded_file is not None and api_key:
        document_content = extract_text_from_pdf(uploaded_file)

        if st.button("Simplify Document"):
            simplified_content, check_results, iterations, initial_metrics, final_metrics, constrained_reasoning_results = simplify_document(document_content, api_key)

            if simplified_content.startswith("Error:") or simplified_content.startswith("Fehler:"):
                st.error(simplified_content)
            else:
                st.subheader("Simplification Iterations")
                for iteration in iterations:
                    with st.expander(f"Iteration {iteration['iteration']}"):
                        st.write(iteration['content'])
                        st.write("Metrics:", iteration['metrics'])

                st.subheader("Final Simplified Text")
                editable_text = st.text_area("Edit the simplified text if needed:", simplified_content, height=300)

                if st.button("Save Edited Text"):
                    with open("simplified_german_document.txt", "w", encoding="utf-8") as f:
                        f.write(editable_text)
                    st.success("Simplified text saved to 'simplified_german_document.txt'")

                if final_metrics:
                    st.subheader("Metrics Comparison")
                    comparison_df = display_metrics_comparison(initial_metrics, final_metrics)
                    st.dataframe(comparison_df)

                if check_results:
                    st.subheader("Hallucination Check Results")
                    st.write(check_results)

                if constrained_reasoning_results:
                    st.subheader("Constrained Reasoning Results")
                    st.write(constrained_reasoning_results)

                st.subheader("Additional Checks")
                coverage_results = coverage_accuracy_assessment(document_content, simplified_content)
                st.write("Coverage Assessment:", coverage_results)

                entity_verification = verify_medical_entities(document_content, simplified_content)
                st.write("Medical Entity Verification:", entity_verification)

                consistency_check = self_consistency_check(document_content, simplified_content, api_key)
                st.write("Self-Consistency Check:", consistency_check)

                citation_check = citation_accuracy_check(document_content, simplified_content)
                st.write("Citation Accuracy Check:", citation_check)

                safeguards = implement_safeguards(simplified_content)
                st.write("Safeguards Implementation:", safeguards)

                two_phase_results = two_phase_approach(document_content, simplified_content)
                st.write("Two-Phase Approach Results:", two_phase_results)

if __name__ == "__main__":
    main()
