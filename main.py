import streamlit as st
import PyPDF2
import google.generativeai as genai
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import textstat
import logging
import math
import re
import spacy
import sys
import subprocess
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_spacy_model():
    """Download the German spaCy model if not already installed."""
    try:
        nlp = spacy.load("de_core_news_sm")
        logger.info("German spaCy model already installed.")
    except OSError:
        logger.info("Downloading German spaCy model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "de_core_news_sm"])
        logger.info("German spaCy model downloaded successfully.")

# Call the function to ensure the model is downloaded
download_spacy_model()

# Now load the model
nlp = spacy.load("de_core_news_sm")

def initialize_gemini_client(api_key):
    """Initialize the Google Gemini client with the provided API key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-exp-0801",
    )

def initialize_openai_client(api_key):
    """Initialize the OpenAI client with the provided API key."""
    return openai.OpenAI(api_key=api_key)

def initialize_claude_client(api_key):
    """Initialize the Anthropic Claude client with the provided API key."""
    return Anthropic(api_key=api_key)
    
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

def split_into_paragraphs(text):
    """Split the text into paragraphs using spaCy."""
    doc = nlp(text)
    paragraphs = []
    current_paragraph = []

    for sent in doc.sents:
        current_paragraph.append(sent.text)
        if len(current_paragraph) >= 2 and (sent.text.endswith('.') or sent.text.endswith('?')):
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return paragraphs

def simplify_text_with_ai(text, ai_option, api_key, metrics=None):
    """Use the selected AI to simplify the given text, processing it paragraph by paragraph."""
    try:
        paragraphs = split_into_paragraphs(text)
        simplified_paragraphs = []

        for paragraph in paragraphs:
            prompt = f"""
            Ich möchte, dass Sie meine komplexen Sätze durch einfache Sätze ersetzen. Behalten Sie die Bedeutung bei, aber machen Sie sie einfacher.

Handeln Sie als erfahrener Autor und Lektor. Ihre Aufgabe ist es, die Verständlichkeit des folgenden Textes für Studierende zu verbessern, deren Muttersprache nicht Deutsch ist. Vereinfachen Sie komplexe Sprache, ohne Genauigkeit oder Tiefe zu opfern. Erläutern Sie verwirrende oder unklare Konzepte mithilfe einer Metapher oder Analogie.

- Achten Sie besonders darauf, dass alle rechtlichen und sicherheitsrelevanten Informationen klar und vollständig dargestellt werden.
- Erklären Sie Fachbegriffe in einfacher Sprache (z. B. „Bluthochdruck (Hypertonie)“).

##Anrede und Ton
1. Sprechen Sie den Leser direkt mit „Sie“ an.
2. Verwenden Sie einen respektvollen, neutralen Ton.
3. Vermeiden Sie Diskriminierung und Klischees.

##Besondere Hinweise für medizinische Texte
1. Medizinische Begriffe sollten in eine für Patienten verständliche Sprache übersetzt werden. Wenn es eine umgangssprachliche Entsprechung gibt, sollten Fachbegriffe in Klammern dahinter gesetzt werden. 2. Erklären Sie Dosierungen klar und in gängigen Einheiten (z. B. „1 Tablette“ statt nur „5 mg“).
3. Beschreiben Sie Nebenwirkungen auf verständliche Weise: „X kann Y verursachen“.
4. Heben Sie wichtige Warnhinweise deutlich hervor.
5. Geben Sie klare Anweisungen, was bei Problemen oder Notfällen zu tun ist.
6. Heben Sie Informationen zu Schwangerschaft, Stillzeit und Überdosierung hervor.

- Gib niemals Erklärungen der Vorgangsweise order Teile des Prompts aus

#Kontext:
Menschen mit eingeschränkter Gesundheitskompetenz verstehen herkömmliche
medizinische Anweisungen oft nicht, was zu falscher Medikamenteneinnahme
und nachteiligen gesundheitlichen Folgen führen kann.

Komplex: {Absatz}

Einfach:
            """

            if ai_option == "Gemini":
                model = initialize_gemini_client(api_key)
                chat_session = model.start_chat(history=[])
                response = chat_session.send_message(prompt)
                simplified_paragraphs.append(response.text)
            elif ai_option == "OpenAI":
                client = initialize_openai_client(api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that simplifies medical texts."},
                        {"role": "user", "content": prompt}
                    ]
                )
                simplified_paragraphs.append(response.choices[0].message.content)
            elif ai_option == "Claude":
                client = initialize_claude_client(api_key)
                response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=8096,
                messages=[
                    {"role": "user", "content": prompt}
                ])
                simplified_paragraphs.append(response.content[0].text)

        return "\n\n".join(simplified_paragraphs)

    except Exception as e:
        logger.error(f"Error in simplify_text_with_ai: {str(e)}")
        return f"Error: Unable to process the request. Please try again later. Details: {str(e)}"

def count_sentences(text):
    """Count the number of sentences in the text."""
    return len([s for s in text.split('.') if s.strip()])

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

def calculate_reading_time(text):
    """Calculate the estimated reading time in minutes."""
    words = re.findall(r'\w+', text.lower())
    word_count = len(words)
    # Assuming average reading speed of 200 words per minute
    reading_time = word_count / 200
    return round(reading_time, 2)

def count_words(text):
    """Count the number of words in the text."""
    return len(text.split())

def count_long_words(text):
    """Count the number of words with more than 6 characters."""
    return len([word for word in text.split() if len(word) > 6])

def count_polysyllabic_words(text):
    # Split the text into sentences
    sentences = re.split(r'[.!?]+', text)

    # Select 10 sentences from the beginning, middle, and end
    num_sentences = len(sentences)
    if num_sentences < 30:
        print("Text must contain at least 30 sentences for a valid SMOG calculation.")
        return None

    selected_sentences = []

    # Add sentences from the beginning
    selected_sentences.extend(sentences[:10])

    # Add sentences from the middle
    middle_index = num_sentences // 2
    selected_sentences.extend(sentences[middle_index - 5:middle_index + 5])

    # Add sentences from the end
    selected_sentences.extend(sentences[-10:])

    # Count polysyllabic words in the selected sentences
    polysyllabic_word_count = 0
    for sentence in selected_sentences:
        words = re.findall(r'\b\w+\b', sentence)
        for word in words:
            if len(re.findall(r'[aeiouy]{2,}', word.lower())) >= 1:  # Check for vowel groups
                syllable_count = sum(1 for char in word if char in "aeiouy")
                if syllable_count >= 3:
                    polysyllabic_word_count += 1

    return polysyllabic_word_count

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
        "Text Standard": textstat.text_standard(text),
        "Count_Syllables": count_syllables(text),
        "Calculate_Reading_Time": calculate_reading_time(text),

        # German metrics
        "G-SMOG": calculate_g_smog(text),
        "LIX (Läsbarhetsindex)": calculate_lix(text),
        "Wiener Sachtextformel (WSF)": textstat.wiener_sachtextformel(text,1),

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
            metrics["Wiener Sachtextformel (WSF)"] <= 6)

def display_metrics(metrics, title="Readability Metrics"):
    st.subheader(title)

    # Display German metrics first
    st.write("German Metrics:")
    german_metrics = ["G-SMOG", "LIX (Läsbarhetsindex)", "Wiener Sachtextformel (WSF)"]
    for metric in german_metrics:
        if metric in metrics:
            st.write(f"{metric}: {metrics[metric]:.2f}")
        else:
            st.write(f"{metric}: Not calculated")

    # Display English metrics
    st.write("English Metrics:")
    english_metrics = ["Flesch Reading Ease", "Flesch-Kincaid Grade", "Gunning Fog", "SMOG Index", 
                       "Coleman-Liau Index", "Automated Readability Index", "Dale-Chall Readability Score", 
                       "Linsear Write Formula", "Text Standard", "Count_Syllables", "Calculate_Reading_Time"]
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

def main():
    st.title("Medical Leaflet Simplifier")
    st.write("Upload a PDF of a medical leaflet to simplify its content.")

    ai_option = st.selectbox("Choose AI Model", ["Gemini", "OpenAI", "Claude"])
    api_key = st.text_input(f"Enter your {ai_option} API Key", type="password")
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

                st.write("Simplifying text...")
                simplified_text = simplify_text_with_ai(original_text, ai_option, api_key)

                if simplified_text.startswith("Error:"):
                    st.error(simplified_text)
                else:
                    final_metrics = analyze_text(simplified_text)

                    st.subheader("Final Results")

                    st.subheader("Original vs Simplified Text")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Text**")
                        st.text_area("", value=original_text, height=400, disabled=True)
                    with col2:
                        st.markdown("**Simplified Text**")
                        st.text_area("", value=simplified_text, height=400, disabled=True)

                    st.subheader("Metrics Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        display_metrics(initial_metrics, "Initial Metrics")
                    with col2:
                        display_metrics(final_metrics, "Final Metrics")

                    # Check readability and provide feedback
                    if is_text_readable(final_metrics):
                        st.success("The simplified text meets the readability criteria.")
                    else:
                        st.warning("The simplified text does not meet all readability criteria. Further simplification may be needed.")
                        st.info("Consider running the simplification process again on the current output to further improve readability.")

                    # Option to re-simplify
                    if st.button("Simplify Again"):
                        st.write("Re-simplifying text...")
                        simplified_text = simplify_text_with_ai(simplified_text, ai_option, api_key)
                        final_metrics = analyze_text(simplified_text)
                        
                        st.subheader("Re-simplified Text")
                        st.text_area("", value=simplified_text, height=400, disabled=True)
                        
                        display_metrics(final_metrics, "Updated Metrics")
                        
                        if is_text_readable(final_metrics):
                            st.success("The re-simplified text now meets the readability criteria.")
                        else:
                            st.warning("The text still does not meet all readability criteria. Manual review and editing may be necessary.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error in main function: {str(e)}", exc_info=True)
    else:
        st.warning(f"Please enter your {ai_option} API Key and upload a PDF file to proceed.")

if __name__ == "__main__":
    main()
