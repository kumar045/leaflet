import streamlit as st
import PyPDF2
import google.generativeai as genai
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import textstat
import logging
import math
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def initialize_openai_client(api_key):
    """Initialize the OpenAI client with the provided API key."""
    return openai.OpenAI(api_key=api_key)

def initialize_claude_client(api_key):
    """Initialize the Anthropic Claude client with the provided API key."""
    return Anthropic(api_key=api_key)

def simplify_text_with_ai(text, ai_option, api_key, metrics=None):
    """Use the selected AI to simplify the given text, optionally using current metrics."""
    try:
        prompt = f"""
      I want you to replace my complex sentences with simple sentences. Keep the meaning the same, but make them simpler. 

Act as an experienced writer and editor. Your task is to improve comprehensibility of the following text for students whose first language is not German. Simplify complex language without sacrificing accuracy or depth. Clarify confusing or unclear concepts using a metaphor or analogy.

- Pay particular attention to ensuring that all legal and safety-related information is presented clearly and completely.

- Explain technical terms in simple language (e.g. "high blood pressure (hypertension)").

##Address and tone
1. Address the reader directly using "Sie".
2. Use a respectful, neutral tone.
3. Avoid discrimination and clichés.

##Special instructions for medical texts
1. Medical terms should be translated into language that patients can understand. If there is a colloquial equivalent, technical terms should be placed in brackets after them.
1. Explain dosages clearly and in common units (e.g. "1 tablet" instead of just "5 mg").
2. Describe side effects in an understandable way: "X can cause Y".
3. Highlight important warnings clearly.
4. Give clear instructions on what to do in case of problems or emergencies.
5. Emphasize information on pregnancy, breastfeeding and overdose.

#Context:
People with limited health literacy often do not understand conventional
medical instructions, which can lead to incorrect medication use
and adverse health consequences.

Complex: On the January 16 episode of Friday Night SmackDown, it was announced that Swagger would defend the ECW title against Hardy in a rematch at the Royal Rumble. 

Simple: In the January 16 Friday Night Smackdown show, they said that Swagger would fight Hardy again to keep the ECW title at the Royal Rumble. 

Complex: Some trails are designated as nature trails, and are used by people learning about the natural world. 

Simple: Some trails are marked as nature trails, and are used by people learning about nature.……

Complex: {text}

Simple:        

"""

        if ai_option == "Gemini":
            model = initialize_gemini_client(api_key)
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text
        elif ai_option == "OpenAI":
            client = initialize_openai_client(api_key)
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that simplifies medical texts."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        elif ai_option == "Claude":
            client = initialize_claude_client(api_key)
            response = client.beta.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=32000,
            messages=[
                {"role": "user", "content": prompt}
            ])
            return response.content[0].text

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

                    iteration = 1
                    max_iterations = 1

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while not is_text_readable(final_metrics) and iteration < max_iterations:
                        status_text.text(f"Iteration {iteration}: Simplifying further...")
                        simplified_text = simplify_text_with_ai(simplified_text, ai_option, api_key, final_metrics)
                        final_metrics = analyze_text(simplified_text)
                        iteration += 1
                        progress_bar.progress(iteration / max_iterations)

                    progress_bar.progress(100)

                    if is_text_readable(final_metrics):
                        status_text.text("The simplified text meets the readability criteria.")
                    else:
                        status_text.text(f"The simplified text still doesn't meet all readability criteria after {max_iterations} iterations.")

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

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error in main function: {str(e)}", exc_info=True)
    else:
        st.warning(f"Please enter your {ai_option} API Key and upload a PDF file to proceed.")

if __name__ == "__main__":
    main()
