import streamlit as st
import PyPDF2
import google.generativeai as genai
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

def simplify_text_with_gemini(text, api_key, metrics=None):
    """Use Google Gemini to simplify the given text, optionally using current metrics."""
    try:
        model = initialize_gemini_client(api_key)
        chat_session = model.start_chat(history=[])

        # Updated prompt with emphasis on formal "Sie" form
        prompt = f"""
        You are an AI assistant specialized in simplifying pharmaceutical and medical instructions. Your task is to rewrite the given text to be easily understood by people with limited health literacy, aiming for a 12-year-old reading level. Follow these guidelines:

        1. Maintain all legal and safety information, including specific instructions for special groups and overdose situations.
        2. Keep the text length similar to the original.
        3. Reduce long sentences, nested clauses, passive voice, nominalizations, long words, multi-syllable words, abstract nouns, and medical jargon.
        4. Include information on use during pregnancy and breastfeeding, and measures for overdose.
        5. Use the following format for the simplified text:
           a. Was ist {{Produkt}} und wofür wird es angewendet?
           b. Was sollten Sie vor der <Einnahme> <Anwendung> von {{Produkt}} beachten?
           c. Wie ist {{Produkt}} <einzunehmen> <anzuwenden>?
           d. Welche Nebenwirkungen sind möglich?
           e. Wie ist {{Produkt}} aufzubewahren?

        6. Aim for these readability indices:
           - G-SMOG: 6 or lower (range: 4-15)
           - Wiener Sachtextformel: 6 or lower (range: 4-15)
           - German LIX: 38 or lower (range: 20-70)

        7. IMPORTANT: Always use the formal "Sie" form in German, not the informal "du" form.

        Before simplifying, consider: What is the gender and cultural background of the target person for these instructions?

        Here are some examples of how to simplify text (note the use of "Sie" form):

        Example 1:
        Original: "Die Applikation erfolgt dreimal täglich mit den Mahlzeiten, um einen stabilen Blutglukosespiegel zu halten."
        Simplified: "Nehmen Sie Ihr Medikament dreimal täglich zu Ihren Mahlzeiten ein. Es hilft, Ihren Zuckerwert stabil zu halten."

        Example 2:
        Original: "Sie sollten daher während des 1 und 2 Schwangerschaftsdrittels {{Produkt}} nur nach Rücksprache mit dem Arzt oder Zahnarzt und nur in der geringsten wirksamen Dosis und für die kürzestmögliche Zeit einnehmen, da es Hinweise auf ein erhöhtes Risiko von Fehlgeburten und Missbildungen gibt."
        Simplified: "Nehmen Sie {{Produkt}} in den ersten 6 Monaten Ihrer Schwangerschaft nur nach Rücksprache mit Ihrem Arzt ein. Bitte nehmen Sie nur die niedrigste Dosis ein. Achten Sie auch darauf {{Produkt}} nur für die kürzest mögliche Zeit einzunehmen. Bei falscher Einnahme kann es zu Fehlgeburten oder Missbildungen bei Ihrem Kind kommen. Weitere Informationen zur Einnahme finden Sie in Kapitel 3 dieser Packungsbeilage."

        Example 3:
        Original: "Vorsicht ist angeraten, wenn Sie gleichzeitig Arzneimittel erhalten, die das Risiko für Geschwüre oder Blutungen erhöhen können, wie z.B. orale Kortikosteroide, blutgerinnungshemmende Medikamente wie Warfarin, selektive Serotonin-Wiederaufnahmehemmer, die unter anderem zur Behandlung von depressiven Verstimmungen eingesetzt werden, oder Thrombozytenaggregationshemmer wie ASS (siehe Abschnitt 2 'Bei Einnahme von {{Produkt}} mit anderen Arzneimitteln')."
        Simplified: "Bitte sprechen Sie mit Ihrem Arzt, wenn Sie {{Produkt}} gleichzeitig mit Medikamenten einnehmen, die:
        - das Risiko für Geschwüre oder Blutungen erhöhen können (z.B. Kortikosteroide zum Schlucken).
        - Medikamente, die die Blutgerinnung hemmen (z.B. Warfarin).
        - eine Aufnahme von Serotonin hemmen (selektive Serotonin-Wiederaufnahmehemmer). Diese werden unter anderem zur Behandlung von Depressionen eingesetzt.
        - Thrombozytenaggregationshemmer wie ASS (weitere Informationen finden Sie unter Abschnitt 2 'Bei Einnahme von {{Produkt}} mit anderen Arzneimitteln')."

        Example 4:
        Original: "Für diese Patienten sowie für Patienten, die eine begleitende Therapie mit niedrig-dosierter Acetylsalicylsäure (ASS) oder anderen Arzneimitteln, die das Risiko für Magen-Darm-Erkrankungen erhöhen können, benötigen, sollte eine Kombinationstherapie mit Magenschleimhaut-schützenden Arzneimitteln (z.B. Misoprostol oder Protonenpumpenhemmer) in Betracht gezogen werden."
        Simplified: "Bitte sprechen Sie mit Ihrem Arzt, wenn Sie:
        - eine begleitende Therapie mit niedrig-dosierter Acetylsalicylsäure (ASS) benötigen.
        - anderen Arzneimitteln einnehmen, die das Risiko für Magen-Darm-Erkrankungen erhöhen können (Beispiel).
        Ihr Arzt wird mit Ihnen besprechen, ob Sie zusätzlich Arzneimittel (z.B. Misoprostol oder Protonenpumpenhemmer) einnehmen sollten, um Ihre Magenschleimhaut zu schützen."

        Example 5:
        Original: "Bei Schmerzen, die länger als 5 Tage anhalten, oder bei Fieber, das länger als 3 Tage anhält oder sich verschlimmert oder wenn weitere Symptome auftreten, sollte ein Arzt aufgesucht werden."
        Simplified: "Bitte suchen Sie Ihren Arzt auf:
        - wenn Ihre Schmerzen länger als 5 Tage anhalten,
        - wenn Ihr Fieber länger als 3 Tage anhält oder sich verschlimmert,
        - wenn weitere Anzeichen von Erkrankungen (Beispiel) auftreten."

        Example 6:
        Original: "Während der Schwangerschaft und Stillzeit darf die empfohlene Dosierung nicht überschritten werden, da eine Überdosierung die Blutversorgung des ungeborenen Kindes beeinträchtigen oder die Milchproduktion vermindern kann."
        Simplified: "Bitte beachten Sie: Überschreiten Sie die Dosierung nicht, wenn Sie schwanger sind oder stillen. Eine Überdosierung kann die Versorgung Ihres ungeborenen Kindes mit Blut beeinträchtigen und die Produktion von Muttermilch vermindern."

        Example 7:
        Original: "Anschließend sollten die Hände gewaschen werden, außer diese wären die zu behandelnde Stelle."
        Simplified: "Bitte waschen Sie anschließend Ihre Hände. Dies gilt nicht, wenn Ihre Hände mit {{Produkt}} behandelt werden."

        Example 8:
        Original: "Von {{Produkt}} soll pro Tag nicht mehr eingenommen werden, als in der Dosierungsanleitung angegeben oder vom Arzt verordnet wurde."
        Simplified: "Nehmen Sie von {{Produkt}} nicht mehr ein, als in der Dosierungsanleitung angegeben wird, oder wie von Ihrem Arzt verordnet. Weitere Information zur Dosierung finden Sie in Kapitel 3 dieser Packungsbeilage."

        Example 9:
        Original: "Bei gleichzeitiger Anwendung von Paracetamol und Zidovudin wird die Neigung zur Verminderung weißer Blutzellen (Neutropenie) verstärkt."
        Simplified: "Wenn Sie Paracetamol gleichzeitig mit Zidovudin anwenden, können sich die Anzahl der weißen Blutzellen in Ihrem Blut vermindern."

        Example 10:
        Original: "Die Anwendung bei chronischem Schnupfen darf wegen der Gefahr des Schwundes der Nasenschleimhaut nur unter ärztlicher Kontrolle erfolgen."
        Simplified: "Wenden Sie {{Produkt}} bei langanhaltendem Schnupfen bitte nur unter ärztlicher Kontrolle an. Ihre Nasenschleimhaut kann beschädigt werden."

        Now, please simplify the following text:

        {text}

        Ensure your simplified version maintains all important information while being more accessible to readers with limited health literacy. Remember to consistently use the formal "Sie" form throughout the text.
        """

        response = chat_session.send_message(prompt)
        
        return response.text
    except Exception as e:
        logger.error("Error in simplify_text_with_gemini: %s", str(e))
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

def calculate_g_smog(text):
    """Calculate the German SMOG (gSMOG) index."""
    sentences = count_sentences(text)
    long_words = count_long_words(text)
    if sentences < 30:
        return 0  # Not enough sentences for accurate calculation
    return round(1.0430 * math.sqrt(30 * long_words / sentences) + 3.1291, 2)

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

                st.write("Simplifying text...")
                simplified_text = simplify_text_with_gemini(original_text, api_key)
                
                if simplified_text.startswith("Error:"):
                    st.error(simplified_text)
                else:
                    final_metrics = analyze_text(simplified_text)

                    iteration = 1
                    max_iterations = 5

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while not is_text_readable(final_metrics) and iteration < max_iterations:
                        status_text.text(f"Iteration {iteration}: Simplifying further...")
                        simplified_text = simplify_text_with_gemini(simplified_text, api_key, final_metrics)
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
        st.warning("Please enter your Google Gemini API Key and upload a PDF file to proceed.")

if __name__ == "__main__":
    main()
