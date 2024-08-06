import streamlit as st
import PyPDF2
import google.generativeai as genai
import re
import math

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def simplify_text_with_gemini(text, api_key, metrics=None):
    """Use Gemini to simplify the given text, optionally using current metrics."""
    genai.configure(api_key=api_key)
    # Create the model
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0801",
    generation_config=generation_config,
    )
    
    base_prompt = """
    Task: Rewrite pharmaceutical and medical instructions to be easily understood by people with limited health literacy, aiming for a 12-year-old reading level. Maintain all legal and safety information, including specific instructions for special groups and overdose situations. Keep the text length similar to the original.

    Before starting, ask: What is the gender and cultural background of the target person for these instructions?

    Target Readability Indices (max values):
    1. G-SMOG: 6 (range: 4-15)
    2. Wiener Sachtextformel: 6 (range: 4-15)
    3. German LIX: 38 (range: 20-70)

    Guidelines:
    - Reduce long sentences, nested clauses, passive voice, nominalizations, long words, multi-syllable words, abstract nouns, and medical jargon
    - Include information on use during pregnancy and breastfeeding, and measures for overdose
    - Maintain all legal and safety information

    Format:
    1. Was ist {Produkt} und wofür wird es angewendet?
    2. Was sollten Sie vor der <Einnahme> <Anwendung> von {Produkt} beachten?
    3. Wie ist {Produkt} <einzunehmen> <anzuwenden>?
    4. Welche Nebenwirkungen sind möglich?
    5. Wie ist {Produkt} aufzubewahren?

    Examples:

    Example 1:
    Original: "Die Applikation erfolgt dreimal täglich mit den Mahlzeiten, um einen stabilen Blutglukosespiegel zu halten."
    Simplified: "Nehmen Sie Ihr Medikament dreimal täglich zu Ihren Mahlzeiten ein. Es hilft, Ihren Zuckerwert stabil zu halten."

    Example 2:
    Original: "Sie sollten daher während des 1 und 2 Schwangerschaftsdrittels {Produkt} nur nach Rücksprache mit dem Arzt oder Zahnarzt und nur in der geringsten wirksamen Dosis und für die kürzestmögliche Zeit einnehmen, da es Hinweise auf ein erhöhtes Risiko von Fehlgeburten und Missbildungen gibt."
    Simplified: "Nehmen Sie {Produkt} in den ersten 6 Monaten Ihrer Schwangerschaft nur nach Rücksprache mit Ihrem Arzt ein. Bitte nehmen Sie nur die niedrigste Dosis ein. Achten Sie auch darauf {Produkt} nur für die kürzest mögliche Zeit einzunehmen. Bei falscher Einnahme kann es zu Fehlgeburten oder Missbildungen bei Ihrem Kind kommen. Weitere Informationen zur Einnahme finden Sie in Kapitel 3 dieser Packungsbeilage."

    Example 3:
    Original: "Vorsicht ist angeraten, wenn Sie gleichzeitig Arzneimittel erhalten, die das Risiko für Geschwüre oder Blutungen erhöhen können, wie z.B. orale Kortikosteroide, blutgerinnungshemmende Medikamente wie Warfarin, selektive Serotonin-Wiederaufnahmehemmer, die unter anderem zur Behandlung von depressiven Verstimmungen eingesetzt werden, oder Thrombozytenaggregationshemmer wie ASS (siehe Abschnitt 2 "Bei Einnahme von {Produkt} mit anderen Arzneimitteln")."
    Simplified: "Bitte sprechen Sie mit Ihrem Arzt, wenn Sie {Produkt} gleichzeitig mit Medikamenten einnehmen, die:
    - das Risiko für Geschwüre oder Blutungen erhöhen können (z.B. Kortikosteroide zum Schlucken).
    - Medikamente, die die Blutgerinnung hemmen (z.B. Warfarin).
    - eine Aufnahme von Serotonin hemmen (selektive Serotonin-Wiederaufnahmehemmer). Diese werden unter anderem zur Behandlung von Depressionen eingesetzt.
    - Thrombozytenaggregationshemmer wie ASS (weitere Informationen finden Sie unter Abschnitt 2 "Bei Einnahme von {Produkt} mit anderen Arzneimitteln")."

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
    Simplified: "Bitte beachten Sie: Überschreiten Sie die Dosierung nicht wenn Sie schwanger sind oder stillen. Eine Überdosierung kann die Versorgung Ihres ungeborenen Kindes mit Blut beeinträchtigen und die Produktion von Muttermilch vermindern."

    Example 7:
    Original: "Anschließend sollten die Hände gewaschen werden, außer diese wären die zu behandelnde Stelle."
    Simplified: "Bitte waschen Sie anschließend Ihre Hände. Dies gilt nicht, wenn Ihre Hände mit {Produkt} behandelt werden."

    Example 8:
    Original: "Von {Produkt} soll pro Tag nicht mehr eingenommen werden, als in der Dosierungsanleitung angegeben oder vom Arzt verordnet wurde."
    Simplified: "Nehmen Sie von {Produkt} nicht mehr ein, als in der Dosierungsanleitung angegeben wird, oder wie von Ihrem Arzt verordnet. Weitere Information zur Dosierung finden Sie in Kapitel 3 dieser Packungsbeilage."

    Example 9:
    Original: "Bei gleichzeitiger Anwendung von Paracetamol und Zidovudin wird die Neigung zur Verminderung weißer Blutzellen (Neutropenie) verstärkt."
    Simplified: "Wenn Sie Paracetamol gleichzeitig mit Zidovudin anwenden, können sich die Anzahl der weißen Blutzellen in Ihrem Blut vermindern."

    Example 10:
    Original: "Die Anwendung bei chronischem Schnupfen darf wegen der Gefahr des Schwundes der Nasenschleimhaut nur unter ärztlicher Kontrolle erfolgen."
    Simplified: "Wenden Sie {Produkt} bei langanhaltendem Schnupfen bitte nur unter ärztlicher Kontrolle an. Ihre Nasenschleimhaut kann beschädigt werden."

    Now, simplify the following text while maintaining its length and all important information:
    """
    
    if metrics:
        metric_feedback = """
        Current readability metrics:
        - G-SMOG: {gsmog:.2f} (target: <= 6)
        - Wiener Sachtextformel: {wstf:.2f} (target: <= 6)
        - German LIX: {lix:.2f} (target: <= 38)
        
        Please adjust the text to improve these metrics while maintaining accuracy and completeness.
        """
        base_prompt += metric_feedback.format(
            gsmog=metrics["G-SMOG"],
            wstf=metrics["Wiener Sachtextformel"],
            lix=metrics["German LIX"]
        )
    
    full_prompt = base_prompt + "\n\nHere's the text to simplify:\n\n" + text
    chat_session = model.start_chat(
    history=[
    ]
    )

    response = chat_session.send_message(full_prompt)
    return response.text

def calculate_gsmog(text):
    sentences = re.split(r'[.!?]+', text)
    word_count = sum(len(re.findall(r'\w+', sentence)) for sentence in sentences)
    polysyllable_count = sum(1 for word in re.findall(r'\w+', text) if count_syllables(word) >= 3)
    
    if word_count >= 30:
        gsmog = 1.0430 * math.sqrt(polysyllable_count * (30 / word_count)) + 3.1291
    else:
        gsmog = 1.0430 * math.sqrt(polysyllable_count * (word_count / 30)) + 3.1291
    
    return gsmog

def calculate_wiener_sachtextformel(text):
    sentences = re.split(r'[.!?]+', text)
    word_count = sum(len(re.findall(r'\w+', sentence)) for sentence in sentences)
    words_with_3plus_syllables = sum(1 for word in re.findall(r'\w+', text) if count_syllables(word) >= 3)
    words_with_6plus_chars = sum(1 for word in re.findall(r'\w+', text) if len(word) > 6)
    
    ms = words_with_3plus_syllables / word_count * 100
    sl = word_count / len(sentences)
    iw = words_with_6plus_chars / word_count * 100
    es = 1 / word_count * 100
    
    wstf = 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875
    
    return wstf

def calculate_german_lix(text):
    sentences = re.split(r'[.!?]+', text)
    word_count = sum(len(re.findall(r'\w+', sentence)) for sentence in sentences)
    words_with_6plus_chars = sum(1 for word in re.findall(r'\w+', text) if len(word) > 6)
    
    average_sentence_length = word_count / len(sentences)
    percentage_long_words = words_with_6plus_chars / word_count * 100
    
    lix = average_sentence_length + percentage_long_words
    
    return lix

def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count

def analyze_text(text):
    """Analyze the readability of the given text."""
    metrics = {
        "G-SMOG": calculate_gsmog(text),
        "Wiener Sachtextformel": calculate_wiener_sachtextformel(text),
        "German LIX": calculate_german_lix(text),
        "Number of sentences": len(re.split(r'[.!?]+', text)),
        "Number of words": len(re.findall(r'\w+', text)),
        "Average words per sentence": len(re.findall(r'\w+', text)) / len(re.split(r'[.!?]+', text)),
    }
    return metrics

def is_text_readable(metrics):
    """Check if the text meets readability criteria."""
    return (metrics["G-SMOG"] <= 6 and
            metrics["Wiener Sachtextformel"] <= 6 and
            metrics["German LIX"] <= 38)

def main():
    st.title("Medical Leaflet Simplifier")
    st.write("Upload a PDF of a medical leaflet to simplify its content.")

    api_key = st.text_input("Enter your Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if api_key and uploaded_file is not None:
        if st.button("Process PDF"):
            st.write("Processing your file...")
            original_text = extract_text_from_pdf(uploaded_file)

            simplified_text = simplify_text_with_gemini(original_text, api_key)
            metrics = analyze_text(simplified_text)

            iteration = 1
            max_iterations = 5

            progress_bar = st.progress(0)
            status_text = st.empty()

            while not is_text_readable(metrics) and iteration < max_iterations:
                status_text.text(f"Iteration {iteration}: Simplifying further...")
                simplified_text = simplify_text_with_gemini(simplified_text, api_key, metrics)
                metrics = analyze_text(simplified_text)
                iteration += 1
                progress_bar.progress(iteration / max_iterations)

            progress_bar.progress(100)
            
            if is_text_readable(metrics):
                status_text.text("The simplified text meets the readability criteria.")
            else:
                status_text.text(f"The simplified text still doesn't meet all readability criteria after {max_iterations} iterations.")

            st.subheader("Original vs Simplified Text")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original vs Simplified Text")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text**")
                st.text_area("", value=original_text, height=400, disabled=True)
            with col2:
                st.markdown("**Simplified Text**")
                st.text_area("", value=simplified_text, height=400, disabled=True)

            st.subheader("Readability Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("G-SMOG", f"{metrics['G-SMOG']:.2f}")
            with col2:
                st.metric("Wiener Sachtextformel", f"{metrics['Wiener Sachtextformel']:.2f}")
            with col3:
                st.metric("German LIX", f"{metrics['German LIX']:.2f}")

            with st.expander("Additional Statistics"):
                st.write(f"Number of sentences: {metrics['Number of sentences']:.0f}")
                st.write(f"Number of words: {metrics['Number of words']:.0f}")
                st.write(f"Average words per sentence: {metrics['Average words per sentence']:.2f}")
    else:
        st.warning("Please enter your Gemini API Key and upload a PDF file to proceed.")

if __name__ == "__main__":
    main()
