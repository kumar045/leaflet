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
from spacy.util import get_package_path

# Function to download and extract the model
def download_model(model_name, version):
    # URL for the model
    url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-{version}/{model_name}-{version}.tar.gz"
    
    # Local filename
    filename = f"{model_name}-{version}.tar.gz"
    
    # Download directory (use /tmp for Streamlit cloud)
    download_dir = ""
    
    # Full path for the downloaded file
    file_path = os.path.join(download_dir, filename)
    
    # Download the file
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the file
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=download_dir)
    
    # Remove the tar.gz file
    os.remove(file_path)
    
    # Return the path to the extracted model
    return os.path.join(download_dir, model_name)

# Download and set up the model
model_name = "de_core_news_sm"
model_version = "3.7.0"  # You may need to update this to the latest version
model_path = download_model(model_name, model_version)

# Add the model path to Python's sys.path
sys.path.append(model_path)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load(model_path)

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
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-exp-0801",
    )

def simplify_text_with_gemini(text, api_key, metrics=None):
    """Use Google Gemini to simplify the given text, optionally using current metrics."""
    try:
        model = initialize_gemini_client(api_key)
        chat_session = model.start_chat(history=[])

        #Updated prompt with emphasis on formal "Sie" form
        prompt = f"""
        You are an AI assistant specialized in simplifying pharmaceutical and medical instructions. Your task is to rewrite the given text to be easily understood by people with limited health literacy, aiming for a 12-year-old reading level. Follow these guidelines:
##Aufgabe: ##
Verfasse Anweisungen zu pharmazeutischen und medizinischen Themen so um, dass sie für Personen mit eingeschränkter Gesundheitskompetenz und einem Leselevel auf dem Niveau eines 14-Jährigen einfach verständlich sind. Achte dabei besonders darauf, dass alle rechtlichen und sicherheitsrelevanten Informationen klar und vollständig präsentiert werden. Dies beinhaltet spezifische Hinweise für spezielle Bevölkerungsgruppen wie Schwangere oder Stillende sowie klare Anweisungen zu Maßnahmen bei einer Überdosierung.

##Spezifika: ##
Reduziere so weit wie möglich
- lange Sätze: maximal 20 Wörter pro Satz

- Schachtelsätze: Keine verschachtelten Sätze, Aufspaltung in mehrere Sätze.

- Nebensätze: maximal zwei Nebensätze

- Nominalisierungen: paraphrasiere

- Passivkonstruktionen: Aktivsätze sind Passivsätzen vorzuziehen

- Partizipialkonstruktionen

- Negationen

- lange Wörter

- mehrsilbige Wörter

- Komposita: verzichten, soferne das nicht die Sinnhaftigkeit beeinträchtigt

- Durchkopplungen (zwei oder mehr Nominale mit einem Nomen zu einer Nominalphrase (z.B. Herr-der-Ringe-Roman) gekoppelt): aus nicht mehr als vier Gliedern

- Fachbegriffe: Medizinische Begriffe sind in eine patientenverständliche Sprache zu übersetzen, Fachausdrücke sind nach einer umgangssprachlichen Entsprechung in Klammern zu platzieren

Beachte
- Direkte Ansprache
Unter einer Appellfunktion wird die persönliche Anrede unter Verwendung von Imperativen verstanden. 
Die Anrede sollte kurz, konkret und anschaulich sein.

- Semantische Zusammenhänge
Um die Plausibilität bei der Beschreibung von Ursache und Wirkung für den Nutzer zu verbessern, muss eine semantische Relation aufgebaut werden muss. Dies wird erreicht, wenn zunächst die Ursache und erst danach die Wirkung genannt wird. Für den Abschnitt „Nebenwirkungen" ist dies von besonderer Bedeutung, da es dem Anwender durch eine immer gleiche Satzstruktur leichter fällt, den Inhalt zu verstehen (und zu befolgen).
Generell werden Formulierungen verständlicher, wenn der Satzaufbau folgendermaßen konstruiert wird:
A kann B verursachen
A kann B bewirken
A ist Ursache von B

-Quantifizierung
Um den Nutzer eine klare Handlungsanweisung vorzugeben, müssen Aussagen quantifiziert werden. Werden Angaben nur vage formuliert, könnten Anweisungen unterschiedlich gedeutet werden.
Angaben sind so zu quantifizieren, dass der Anwender diese direkt versteht. Eine Dosierungsangabe zum Beispiel nur in Milligramm anzugeben, würde dem wiedersprechen, da dies vom Patienten verlangen würde, die benötigte Stückzahl selbst zu berechnen.


Berücksichtige unbedingt Hinweise zur Anwendung während der Schwangerschaft und Stillzeit sowie zu Maßnahmen bei Überdosierung.

Beispiele für lange Wörter:
- Nasenschleimhautschwellungen
- Selektive-Serotonin-Wiederaufnahme-Hemmer
- Thrombozytenaggregationshemmern
- Nasenatmungsbehinderung

Beispiele abstrakte Substantive:
- Absetzsymptomatik
- Eisenmangelanämie
- Grundkrankheit
- Anstaltspackung

Bearbeite die folgenden Punkte und alle jeweiligen Unterpunkte Schritt für Schritt genau in der vorgegebenen Reihenfolge und nimm dir ausreichend Zeit dafür, um es korrekt zu machen.
Überprüfe zwischen jedem Schritt, ob alle Informationen vollständig vorhanden sind.

1. Nebenwirkungen
2. Wechselwirkungen
3. Einnahmehinweise
4. Sonstige Hinweise

##Kontext: ##
Personen mit begrenzter Gesundheitskompetenz verstehen oft herkömmliche medizinische Anweisungen nicht, was zu fehlerhafter Medikamentenanwendung und ungünstigen Gesundheitsfolgen führen kann.

##Beispiele:##


Beispiel 1:

**Originaltext:**
"Hohe Dosen sollten ohne Rücksprache mit dem Arzt über längere Zeiträume nicht eingenommen werden."

**Vereinfachter Text:**
"Dosen von mehr als 3 mal 1 Tabl. pro Tag sollten Sie ohne Rücksprache mit Ihrem Arzt nicht länger als 5 Tage einnehmen."


Beispiel 2:
Lange Sätze

**Originaltext:**
"Sie sollten daher während des 1 und 2 Schwangerschaftsdrittels {{Produkt}}  nur nach Rücksprache mit dem Arzt oder Zahnarzt und nur in der geringsten wirksamen Dosis und für die kürzestmögliche Zeit einnehmen, da es Hinweise auf ein erhöhtes Risiko von Fehlgeburten und Missbildungen gibt."

**Vereinfachter Text:**
"Nehmen Sie Produkt in den ersten 6 Monaten Ihrer Schwangerschaft  nur nach Rücksprache mit Ihrem Arzt ein. Bitte nehmen Sie nur die niedrigste Dosis ein. Achten Sie auch darauf {{Produkt}} nur für die kürzest mögliche Zeit einzunehmen. Bei falscher Einnahm kann es zu Fehlgeburten oder Missbildungen bei Ihrem Kind kommen. Weitere Informationen zur Einnahme finden Sie in Kapitel 3 dieser Packungsbeilage."

Beispiel 3:
Lange Sätze

**Originaltext:**
"Vorsicht ist angeraten, wenn Sie gleichzeitig Arzneimittel erhalten, die das Risiko für Geschwüre oder Blutungen erhöhen können, wie z.B. orale Kortikosteroide, blutgerinnungshemmende Medikamente wie Warfarin, selektive Serotonin-Wiederaufnahmehemmer, die unter anderem zur Behandlung von depressiven Verstimmungen eingesetzt werden, oder Thrombozytenaggregationshemmer wie ASS (siehe Abschnitt 2 "Bei Einnahme von {{Produkt}} mit anderen Arzneimitteln")."

**Vereinfachter Text:**
"Bitte sprechen Sie mit Ihrem Arzt, wenn Sie {{Produkt}} gleichzeitig mit Medikamenten einnehmen, die:
- das Risiko für Geschwüre oder Blutungen erhöhen können (z.B. Kortikosteroide zum Schlucken).
- Medikamente, die die Blutgerinnung hemmen (z.B. Warfarin).
- eine Aufnahme von Serotonin hemmen (selektive SerotoninWiederaufnahmehemmer). Diese werden unter anderem zur Behandlung von Depressionen eingesetzt.
- Thrombozytenaggregationshemmer wie ASS (weitere Informationen finden Sie unter Abschnitt 2 "Bei Einnahme von {{Produkt}} mit anderen Arzneimitteln")."

Beispiel 4:
Schachtelsätze

**Originaltext:**
"Für diese Patienten sowie für Patienten, die eine begleitende Therapie mit niedrig-dosierter Acetylsalicylsäure (ASS) oder anderen Arzneimitteln, die das Risiko für Magen-Darm-Erkrankungen erhöhen können, benötigen, sollte eine Kombinationstherapie mit Magenschleimhaut-schützenden Arzneimitteln (z.B. Misoprostol oder Protonenpumpenhemmer) in Betracht gezogen werden."

**Vereinfachter Text:**
"Bitte sprechen Sie mit Ihrem Arzt, wenn Sie:
- eine begleitende Therapie mit niedrig-dosierter Acetylsalicylsäure (ASS) benötigen.
- anderen Arzneimitteln einnehmen, die das Risiko für Magen-DarmErkrankungen erhöhen können (Beispiel).
Ihr Arzt wird mit Ihnen besprechen, ob Sie zusätzlich Arzneimittel (z.B. Misoprostol oder Protonenpumpenhemmer) einnehmen sollten, um Ihre Magenschleimhaut zu schützen."


Beispiel 5:
Schachtelsätze

**Originaltext:**
"Bei Schmerzen, die länger als 5 Tage anhalten, oder bei Fieber, das länger als 3 Tage anhält oder sich verschlimmert oder wenn weitere Symptome auftreten, sollte ein Arzt aufgesucht werden."

**Vereinfachter Text:**
"Bitte suchen Sie Ihren Arzt auf:
- wenn Ihre Schmerzen länger als 5 Tage anhalten,
- wenn Ihr Fieber länger als 3 Tage anhält oder sich verschlimmert,
- wenn weitere Anzeichen von Erkrankungen (Beispiel) auftreten."


Beispiel 6:
Schachtelsätze

**Originaltext:**
"{{Produkt}}, das, nachdem es eingenommen wurde, Übelkeit verursachen kann, sollte nicht abgesetzt werden, da Nebenwirkungen nach wenigen Tagen verschwinden."

**Vereinfachter Text:**
"Das Produkt kann in den ersten Tagen der Einnahme Übelkeit verursachen. Setzen Sie das Arzneimittel nicht ab, da diese Nebenwirkung in der Regel nach wenigen Tagen verschwindet."


Beispiel 7:
Nominalisierungen

**Originaltext:**
"Das Einnehmen von weiteren Medikamenten muss mit ihren Arzt abgesprochen sein."

**Vereinfachter Text:**
"Sprechen Sie mit ihrem Arzt, bevor Sie weitere Medikamente einnehmen."


Beispiel 8:
Nominalisierungen

**Originaltext:**
"Bei gleichzeitiger Anwendung von Paracetamol und Zidovudin wird die Neigung zur Verminderung weißer Blutzellen (Neutropenie) verstärkt."

**Vereinfachter Text:**
"Wenn Sie Paracetamol gleichzeitig mit Zidovudin anwenden, können sich die Anzahl der weißen Blutzellen in Ihrem Blut vermindern."

Beispiel 9:
Nominalisierungen

**Originaltext:**
"Die Anwendung bei chronischem Schnupfen darf wegen der Gefahr des Schwundes der Nasenschleimhaut nur unter ärztlicher Kontrolle erfolgen."

**Vereinfachter Text:**
"Wenden Sie Produkt bei langanhaltendem Schnupfen bitte nur unter ärztlicher Kontrolle an. Ihre Nasenschleimhaut kann beschädigt werden."


Beispiel 10:
Passivsätze

**Originaltext:**
"Während der Schwangerschaft und Stillzeit darf die empfohlene Dosierung nicht überschritten werden, da eine Überdosierung die Blutversorgung des ungeborenen Kindes beeinträchtigen oder die Milchproduktion vermindern kann."

**Vereinfachter Text:**
"Bitte beachten Sie: Überschreiten Sie die Dosierung nicht wenn Sie schwanger sind oder stillen. Eine Überdosierung kann die Versorgung Ihres ungeborenen Kindes mit Blut beeinträchtigen und die Produktion von Muttermilch vermindern."


Beispiel 11:
Passivsätze

**Originaltext:**
"Anschließend sollten die Hände gewaschen werden, außer diese wären die zu behandelnde Stelle."

**Vereinfachter Text:**
"Bitte waschen Sie anschließend Ihre Hände. Dies gilt nicht, wenn Ihre Hände mit {{Produkt}} behandelt werden."


Beispiel 12:
Passivsätze

**Originaltext:**
"Von Produkt soll pro Tag nicht mehr eingenommen werden, als in der Dosierungsanleitung angegeben oder vom Arzt verordnet wurde."

**Vereinfachter Text:**
"Nehmen Sie von {{Produkt}} nicht mehr ein, als in der Dosierungsanleitung angegeben wird, oder wie von Ihrem Arzt verordnet. Weitere Information zur Dosierung finden Sie in Kapitel 3 dieser Packungsbeilage." 


Beispiel 13:
Partizipialkonstruktionen

**Originaltext:**
"Übergehend in die Muttermilch stellt {{Produkt}} für ein Neugeborenes eine Gefahr dar."

**Vereinfachter Text:**
"Produkt stellt für ein Neugeborenes eine Gefahr dar, weil es in die Muttermilch übergeht."


Beispiel 14:
Komposita

**Originaltext:**
"Nebenschilddrüsenunterfunktion"

**Vereinfachter Text:**
"Unterfunktion der Nebenschilddrüsen"
*Achtung: Sinnhaftigkeit beeinträchtigt:*
Mutterkuchen
Kuchen der Mutter

Beispiel 15:
Fachbegriffe

**Originaltext:**
"Produkt darf bei einem apoplektischen Insult nicht eingenommen werden."

**Vereinfachter Text:**
"Produkt darf nicht eingenommen werden bei einem plötzlich auftreten- den Schlaganfall (apoplektischer Insult)."


Beispiel 16:
Direkte Ansprache

**Originaltext:**
"Es sollen zwei Tabletten eingenommen werden."

**Vereinfachter Text:**
"Nehmen Sie zwei Tabletten."

Beispiel 17a:
Semantische Zusammenhänge

**Originaltext:**
"Wenn gastrointestinale Nebenwirkungen auftreten, dann ist das {{Produkt}} abzusetzen"

**Vereinfachter Text:**
"Produkt kann gastrointestinale Nebenwirkungen auslösen. Es muss dann abgesetzt werden."

Beispiel 17b:
Semantische Zusammenhänge - Apellfunktion

**Originaltext:**
"Wenn gastrointestinale Nebenwirkungen auftreten, dann ist das {{Produkt}} abzusetzen"

**Vereinfachter Text:**
"Produkt kann gastrointestinale Nebenwirkungen auslösen. Setzen Sie das Präparat ab, wenn Sie solche Nebenwirkungen spüren."


Beispiel 18:
Quantifizierung

**Originaltext:**
"Es sollen zwei Tabletten eingenommen werden."

**Vereinfachter Text:**
"Nehmen Sie zwei Tabletten."



##Ausgabeformat:##

##Produkt

##Nebenwirkungen##

##Wechselwirkungen##

##Einnahmehinweise##

##Sonstige Hinweise##


##Hinweise:##
Stelle sicher, dass alle rechtlichen und Sicherheitsinformationen in der vereinfachten Version erhalten bleiben.

        Now, please simplify the following text:

        {text}

        Ensure your simplified version maintains all important information while being more accessible to readers with limited health literacy. Remember to consistently use the formal "Sie" form throughout the text.
        """

        if metrics:
            prompt += f"\n\nCurrent readability metrics: {metrics}\nPlease improve these metrics in your simplification."

        response = chat_session.send_message(prompt)

        return response.text
    except Exception as e:
        logger.error("Error in simplify_text_with_gemini: %s", str(e))
        return f"Error: Unable to process the request. Please try again later. Details: {str(e)}"

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="semantic_similarity"
        )
        embeddings.append(result['embedding'])
    return embeddings

def cosine_similarity(embeddings1, embeddings2):
    similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            similarity_matrix[i, j] = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity_matrix

def coverage_accuracy_assessment(original_text, simplified_text):
    original_sentences = [sent.text.strip() for sent in nlp(original_text).sents]
    simplified_sentences = [sent.text.strip() for sent in nlp(simplified_text).sents]

    original_embeddings = get_embeddings(original_sentences)
    simplified_embeddings = get_embeddings(simplified_sentences)

    similarity_matrix = cosine_similarity(original_embeddings, simplified_embeddings)

    covered_sentences = sum(similarity_matrix.max(axis=1) > 0.8)  # Threshold can be adjusted
    coverage_score = covered_sentences / len(original_sentences)

    return {
        'coverage_score': coverage_score,
        'covered_sentences': covered_sentences,
        'total_original_sentences': len(original_sentences)
    }

def verify_medical_entities(original_text, simplified_text):
    """Verify that medical entities in the original text are preserved in the simplified text."""
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

def self_consistency_check(original_text, simplified_text, api_key, num_versions=3):
    """Generate multiple simplified versions and compare them for consistency."""
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
    """Check if citations in the original text are preserved in the simplified text."""
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
    """Implement safeguards and guardrails to ensure critical information is preserved."""
    safeguards = [
        ("dosage", r"\b\d+\s*(mg|g|ml)\b"),
        ("warning", r"\b(warning|caution|alert)\b"),
        ("side[- ]effects?", r"\bside[- ]effects?\b"),
    ]

    results = {}
    for name, pattern in safeguards:
        if re.search(pattern, simplified_text, re.IGNORECASE):
            results[name] = "Present"
        else:
            results[name] = "Missing"

    return results

def two_phase_approach(original_text, simplified_text):
    """Implement a two-phase approach inspired by the KnowHalu framework."""
    original_key_phrases = set(re.findall(r'\b\w+(?:\s+\w+){2,3}\b', original_text))
    simplified_key_phrases = set(re.findall(r'\b\w+(?:\s+\w+){2,3}\b', simplified_text))
    
    non_fabrication_score = len(original_key_phrases.intersection(simplified_key_phrases)) / len(original_key_phrases)

    original_entities = set(ent.text for ent in nlp(original_text).ents)
    simplified_entities = set(ent.text for ent in nlp(simplified_text).ents)
    
    factual_accuracy_score = len(original_entities.intersection(simplified_entities)) / len(original_entities)

    return {
        'non_fabrication_score': non_fabrication_score,
        'factual_accuracy_score': factual_accuracy_score
    }

def factuality_faithfulness_check(original_text, simplified_text):
    """Check factuality and faithfulness of the simplified text."""
    original_doc = nlp(original_text)
    simplified_doc = nlp(simplified_text)

    original_entities = set(ent.text for ent in original_doc.ents)
    simplified_entities = set(ent.text for ent in simplified_doc.ents)
    factuality_score = len(original_entities.intersection(simplified_entities)) / len(original_entities)

    original_sentences = [sent.text for sent in original_doc.sents]
    simplified_sentences = [sent.text for sent in simplified_doc.sents]
    
    original_embeddings = get_embeddings(original_sentences)
    simplified_embeddings = get_embeddings(simplified_sentences)
    
    similarities = cosine_similarity(original_embeddings, simplified_embeddings)
    faithfulness_score = similarities.max(axis=1).mean()

    return {
        'factuality_score': factuality_score,
        'faithfulness_score': faithfulness_score
    }

def count_syllables(word):
    """Count the number of syllables in a word."""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def calculate_g_smog(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s.]', '', text)

    # Split into sentences
    sentences = re.split(r'\.+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    number_of_sentences = len(sentences)

    # Count words with three or more syllables
    words = text.split()
    words_with_three_or_more_syllables = sum(1 for word in words if count_syllables(word) >= 3)

    # Calculate gSmog
    if number_of_sentences == 0:
        return 0  # Avoid division by zero

    gsmog = math.sqrt((words_with_three_or_more_syllables * 30) / number_of_sentences) - 2
    return gsmog

def calculate_lix(text):
    """Calculate the LIX readability score."""
    words = re.findall(r'\w+', text.lower())
    sentences = re.findall(r'\w+[.!?]', text)

    total_words = len(words)
    total_sentences = len(sentences)
    long_words = sum(1 for word in words if len(word) > 6)

    if total_sentences == 0:
        return 0

    average_sentence_length = total_words / total_sentences
    percentage_long_words = (long_words / total_words) * 100

    lix = average_sentence_length + percentage_long_words
    return round(lix, 2)

def analyze_text(text):
    """Analyze the readability of the given text using both English and German metrics."""
    metrics = {
        # English metrics
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "SMOG Index": textstat.smog_index(text),
        "Coleman-Liau Index": textstat.coleman_liau_index(text),
        "Automated Readability Index": textstat.automated_readability_index(text),
        "Dale-Chall Readability Score": textstat.dale_chall_readability_score(text),
        "Linsear Write Formula": textstat.linsear_write_formula(text),

        # German metrics
        "G-SMOG": calculate_g_smog(text),
        "LIX (Läsbarhetsindex)": calculate_lix(text),
        "Wiener Sachtextformel": textstat.wiener_sachtextformel(text, 1),

        # General statistics
        "Number of sentences": textstat.sentence_count(text),
        "Number of words": textstat.lexicon_count(text),
        "Number of complex words": textstat.difficult_words(text),
        "Percentage of complex words": (textstat.difficult_words(text) / textstat.lexicon_count(text) * 100) if textstat.lexicon_count(text) > 0 else 0,
        "Average words per sentence": textstat.avg_sentence_length(text),
        "Average syllables per word": textstat.avg_syllables_per_word(text)
    }
    return metrics

def is_text_readable(metrics):
    """Check if the text meets readability criteria for both English and German metrics."""
    return (60 <= metrics["Flesch Reading Ease"] <= 70 and
            6 <= metrics["Flesch-Kincaid Grade"] <= 8 and
            8 <= metrics["Gunning Fog"] <= 10 and
            7 <= metrics["SMOG Index"] <= 9 and
            7 <= metrics["Coleman-Liau Index"] <= 9 and
            7 <= metrics["Automated Readability Index"] <= 9 and
            metrics["G-SMOG"] <= 6 and
            metrics["LIX (Läsbarhetsindex)"] <= 38 and
            metrics["Wiener Sachtextformel"] <= 6)

def display_metrics(metrics, title="Readability Metrics"):
    st.subheader(title)

    # Display German metrics first
    st.write("German Metrics:")
    german_metrics = ["G-SMOG", "LIX (Läsbarhetsindex)", "Wiener Sachtextformel"]
    for metric in german_metrics:
        if metric in metrics:
            st.write(f"{metric}: {metrics[metric]:.2f}")
        else:
            st.write(f"{metric}: Not calculated")

    # Display English metrics
    st.write("English Metrics:")
    english_metrics = ["Flesch Reading Ease", "Flesch-Kincaid Grade", "Gunning Fog", "SMOG Index", 
                       "Coleman-Liau Index", "Automated Readability Index", "Dale-Chall Readability Score", 
                       "Linsear Write Formula"]
    for metric in english_metrics:
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, (int, float)):
                st.write(f"{metric}: {value:.2f}")
            else:
                st.write(f"{metric}: {value}")
        else:
            st.write(f"{metric}: Not calculated")

    # Display general statistics
    st.write("General Statistics:")
    general_stats = ["Number of sentences", "Number of words", "Number of complex words", 
                     "Percentage of complex words", "Average words per sentence", "Average syllables per word"]
    for metric in general_stats:
        if metric in metrics:
            st.write(f"{metric}: {metrics[metric]:.2f}")
        else:
            st.write(f"{metric}: Not calculated")

def process_text_with_hallucination_checks(original_text, api_key):
    simplified_text = simplify_text_with_gemini(original_text, api_key)
    
    iterations = []
    max_iterations = 5
    
    for i in range(max_iterations):
        entity_verification = verify_medical_entities(original_text, simplified_text)
        consistency_check = self_consistency_check(original_text, simplified_text, api_key)
        citation_check = citation_accuracy_check(original_text, simplified_text)
        coverage_assessment = coverage_accuracy_assessment(original_text, simplified_text)
        safeguards = implement_safeguards(simplified_text)
        two_phase_results = two_phase_approach(original_text, simplified_text)
        factuality_faithfulness = factuality_faithfulness_check(original_text, simplified_text)
        
        metrics = analyze_text(simplified_text)
        
        iteration_results = {
            'iteration': i + 1,
            'simplified_text': simplified_text,
            'entity_verification': entity_verification,
            'consistency_check': consistency_check,
            'citation_check': citation_check,
            'coverage_assessment': coverage_assessment,
            'safeguards': safeguards,
            'two_phase_results': two_phase_results,
            'factuality_faithfulness': factuality_faithfulness,
            'metrics': metrics
        }
        
        iterations.append(iteration_results)
        
        if (entity_verification['preservation_score'] > 0.9 and
            consistency_check['consistency_score'] > 0.8 and
            citation_check['citation_accuracy'] > 0.9 and
            coverage_assessment['coverage_score'] > 0.8 and
            all(v == "Present" for v in safeguards.values()) and
            two_phase_results['non_fabrication_score'] > 0.8 and
            two_phase_results['factual_accuracy_score'] > 0.8 and
            factuality_faithfulness['factuality_score'] > 0.8 and
            factuality_faithfulness['faithfulness_score'] > 0.8 and
            is_text_readable(metrics)):
            break
        
        simplified_text = simplify_text_with_gemini(simplified_text, api_key)
    
    return simplified_text, iterations

def display_iteration_metrics(iterations):
    st.subheader("Metrics Comparison Across Iterations")
    
    metrics_to_compare = [
        "Flesch Reading Ease", "Flesch-Kincaid Grade", "Gunning Fog", "SMOG Index",
        "Coleman-Liau Index", "Automated Readability Index", "G-SMOG", "LIX (Läsbarhetsindex)",
        "Wiener Sachtextformel", "Number of sentences", "Number of words", "Number of complex words",
        "Percentage of complex words", "Average words per sentence", "Average syllables per word"
    ]
    
    df = pd.DataFrame({
        f"Iteration {i['iteration']}": [i['metrics'][m] for m in metrics_to_compare]
        for i in iterations
    }, index=metrics_to_compare)
    
    st.dataframe(df.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='lightcoral'))

def main():
    st.title("Medical Leaflet Simplifier with Hallucination Detection")
    st.write("Upload a PDF of a medical leaflet to simplify its content and detect potential hallucinations.")

    api_key = st.text_input("Enter your Google Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if api_key and uploaded_file is not None:
        if st.button("Process PDF"):
            try:
                st.write("Processing your file...")
                original_text = extract_text_from_pdf(uploaded_file)

                if not original_text:
                    st.error("Failed to extract text from the PDF. Please check if the file is readable.")
                    return

                st.subheader("Original Text")
                st.text_area("", value=original_text, height=200, disabled=True)

                initial_metrics = analyze_text(original_text)

                st.subheader("Initial Metrics")
                display_metrics(initial_metrics)

                st.write("Simplifying text and performing hallucination checks...")
                simplified_text, iterations = process_text_with_hallucination_checks(original_text, api_key)

                st.subheader("Simplification and Hallucination Check Results")
                for i, iteration in enumerate(iterations):
                    st.write(f"Iteration {i + 1}")
                    st.write(f"Entity Preservation Score: {iteration['entity_verification']['preservation_score']:.2f}")
                    st.write(f"Consistency Score: {iteration['consistency_check']['consistency_score']:.2f}")
                    st.write(f"Citation Accuracy: {iteration['citation_check']['citation_accuracy']:.2f}")
                    st.write(f"Coverage Score: {iteration['coverage_assessment']['coverage_score']:.2f}")
                    st.write(f"Safeguards: {iteration['safeguards']}")
                    st.write(f"Non-fabrication Score: {iteration['two_phase_results']['non_fabrication_score']:.2f}")
                    st.write(f"Factual Accuracy Score: {iteration['two_phase_results']['factual_accuracy_score']:.2f}")
                    st.write(f"Factuality Score: {iteration['factuality_faithfulness']['factuality_score']:.2f}")
                    st.write(f"Faithfulness Score: {iteration['factuality_faithfulness']['faithfulness_score']:.2f}")
                    st.write("---")

                display_iteration_metrics(iterations)

                final_metrics = analyze_text(simplified_text)

                st.subheader("Final Results")

                st.subheader("Original vs Simplified Text")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Text**")
                    st.text_area("", value=original_text, height=400, disabled=True)
                with col2:
                    st.markdown("**Simplified Text**")
                    edited_simplified_text = st.text_area("Edit if needed:", value=simplified_text, height=400)

                st.subheader("Metrics Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    display_metrics(initial_metrics, "Initial Metrics")
                with col2:
                    display_metrics(final_metrics, "Final Metrics")

                if is_text_readable(final_metrics):
                    st.success("The simplified text meets the readability criteria.")
                else:
                    st.warning("The simplified text still doesn't meet all readability criteria.")

                if st.button("Save Edited Text"):
                    st.session_state.saved_simplified_text = edited_simplified_text
                    st.success("Edited text saved successfully!")

                if 'saved_simplified_text' in st.session_state:
                    st.subheader("Saved Simplified Text")
                    st.text_area("", value=st.session_state.saved_simplified_text, height=400, disabled=True)

                if st.button("Re-analyze Edited Text"):
                    edited_text_to_analyze = st.session_state.get('saved_simplified_text', edited_simplified_text)
                    edited_metrics = analyze_text(edited_text_to_analyze)
                    st.subheader("Edited Text Metrics")
                    display_metrics(edited_metrics, "Edited Text Metrics")

                    edited_entity_verification = verify_medical_entities(original_text, edited_text_to_analyze)
                    edited_consistency_check = self_consistency_check(original_text, edited_text_to_analyze, api_key)
                    edited_citation_check = citation_accuracy_check(original_text, edited_text_to_analyze)
                    edited_coverage_assessment = coverage_accuracy_assessment(original_text, edited_text_to_analyze)
                    edited_safeguards = implement_safeguards(edited_text_to_analyze)
                    edited_two_phase_results = two_phase_approach(original_text, edited_text_to_analyze)
                    edited_factuality_faithfulness = factuality_faithfulness_check(original_text, edited_text_to_analyze)

                    st.subheader("Hallucination Check Results for Edited Text")
                    st.write(f"Entity Preservation Score: {edited_entity_verification['preservation_score']:.2f}")
                    st.write(f"Consistency Score: {edited_consistency_check['consistency_score']:.2f}")
                    st.write(f"Citation Accuracy: {edited_citation_check['citation_accuracy']:.2f}")
                    st.write(f"Coverage Score: {edited_coverage_assessment['coverage_score']:.2f}")
                    st.write(f"Safeguards: {edited_safeguards}")
                    st.write(f"Non-fabrication Score: {edited_two_phase_results['non_fabrication_score']:.2f}")
                    st.write(f"Factual Accuracy Score: {edited_two_phase_results['factual_accuracy_score']:.2f}")
                    st.write(f"Factuality Score: {edited_factuality_faithfulness['factuality_score']:.2f}")
                    st.write(f"Faithfulness Score: {edited_factuality_faithfulness['faithfulness_score']:.2f}")

                    if is_text_readable(edited_metrics):
                        st.success("The edited text meets the readability criteria.")
                    else:
                        st.warning("The edited text still doesn't meet all readability criteria.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error in main function: {str(e)}", exc_info=True)
    else:
        st.warning("Please enter your Google Gemini API Key and upload a PDF file to proceed.")

if __name__ == "__main__":
    main()
