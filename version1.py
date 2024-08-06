import streamlit as st
import PyPDF2
import anthropic
import textstat
import logging
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def simplify_text_with_claude(text, api_key, metrics=None):
    """Use Claude to simplify the given text, optionally using current metrics."""
    try:
        client = Anthropic(api_key=api_key)
        
        base_prompt = """
        {human_prompt} You are an AI assistant specialized in simplifying pharmaceutical and medical instructions. Your task is to rewrite the given text to be easily understood by people with limited health literacy, aiming for a 12-year-old reading level. Follow these guidelines:

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

        Before simplifying, consider: What is the gender and cultural background of the target person for these instructions?

        Here are examples of simplification:

        Example 1:
        Original: "Die Applikation erfolgt dreimal täglich mit den Mahlzeiten, um einen stabilen Blutglukosespiegel zu halten."
        Simplified: "Nehmen Sie Ihr Medikament dreimal täglich zu Ihren Mahlzeiten ein. Es hilft, Ihren Zuckerwert stabil zu halten."

        Example 2:
        Original: "Sie sollten daher während des 1 und 2 Schwangerschaftsdrittels {{Produkt}} nur nach Rücksprache mit dem Arzt oder Zahnarzt und nur in der geringsten wirksamen Dosis und für die kürzestmögliche Zeit einnehmen, da es Hinweise auf ein erhöhtes Risiko von Fehlgeburten und Missbildungen gibt."
        Simplified: "Nehmen Sie {{Produkt}} in den ersten 6 Monaten Ihrer Schwangerschaft nur nach Rücksprache mit Ihrem Arzt ein. Bitte nehmen Sie nur die niedrigste Dosis ein. Achten Sie auch darauf {{Produkt}} nur für die kürzest mögliche Zeit einzunehmen. Bei falscher Einnahme kann es zu Fehlgeburten oder Missbildungen bei Ihrem Kind kommen. Weitere Informationen zur Einnahme finden Sie in Kapitel 3 dieser Packungsbeilage."

        Now, please simplify the following text:

        {text_to_simplify}

        Ensure your simplified version maintains all important information while being more accessible to readers with limited health literacy.

        {ai_prompt}
        """
        
        base_prompt = base_prompt.format(
            human_prompt=HUMAN_PROMPT,
            text_to_simplify=text,
            ai_prompt=AI_PROMPT
        )
        
        if metrics:
            metric_feedback = """
            Current readability metrics:
            - Flesch Reading Ease: {flesch_ease:.2f} (target: 60-70)
            - Flesch-Kincaid Grade: {flesch_kincaid:.2f} (target: 6-8)
            - Gunning Fog: {gunning_fog:.2f} (target: 8-10)
            - SMOG Index: {smog:.2f} (target: 7-9)
            - Coleman-Liau Index: {coleman_liau:.2f} (target: 7-9)
            - Automated Readability Index: {ari:.2f} (target: 7-9)
            
            Please adjust the text to improve these metrics while maintaining accuracy and completeness.

            {human_prompt} Using the metrics provided above, please simplify the text further to improve readability while maintaining all important information. {ai_prompt}
            """
            metric_feedback = metric_feedback.format(
                flesch_ease=metrics['Flesch Reading Ease'],
                flesch_kincaid=metrics['Flesch-Kincaid Grade'],
                gunning_fog=metrics['Gunning Fog'],
                smog=metrics['SMOG Index'],
                coleman_liau=metrics['Coleman-Liau Index'],
                ari=metrics['Automated Readability Index'],
                human_prompt=HUMAN_PROMPT,
                ai_prompt=AI_PROMPT
            )
            base_prompt += metric_feedback

        try:
            response = client.completions.create(
                model="claude-2.1",
                max_tokens_to_sample=2000,
                prompt=base_prompt
            )
            return response.completion
        except Exception as e:
            logger.error("API call failed: %s", str(e))
            return "Error: Unable to generate content. Please check your API key and try again. Details: {}".format(str(e))
    except Exception as e:
        logger.error("Error in simplify_text_with_claude: %s", str(e))
        return "Error: Unable to process the request. Please try again later. Details: {}".format(str(e))

def analyze_text(text):
    """Analyze the readability of the given text."""
    metrics = {
        "Flesch Reading Ease": textstat.flesch_reading_ease(text),
        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(text),
        "Gunning Fog": textstat.gunning_fog(text),
        "SMOG Index": textstat.smog_index(text),
        "Coleman-Liau Index": textstat.coleman_liau_index(text),
        "Automated Readability Index": textstat.automated_readability_index(text),
        "Dale-Chall Readability Score": textstat.dale_chall_readability_score(text),
        "Linsear Write Formula": textstat.linsear_write_formula(text),
        "Text Standard": textstat.text_standard(text),
        "Number of sentences": textstat.sentence_count(text),
        "Number of words": textstat.lexicon_count(text),
        "Number of complex words": textstat.difficult_words(text),
        "Percentage of complex words": (textstat.difficult_words(text) / textstat.lexicon_count(text) * 100),
        "Average words per sentence": textstat.avg_sentence_length(text),
        "Average syllables per word": textstat.avg_syllables_per_word(text)
    }
    return metrics

def is_text_readable(metrics):
    """Check if the text meets readability criteria."""
    return (60 <= metrics["Flesch Reading Ease"] <= 70 and
            6 <= metrics["Flesch-Kincaid Grade"] <= 8 and
            8 <= metrics["Gunning Fog"] <= 10 and
            7 <= metrics["SMOG Index"] <= 9 and
            7 <= metrics["Coleman-Liau Index"] <= 9 and
            7 <= metrics["Automated Readability Index"] <= 9)

def main():
    st.title("Medical Leaflet Simplifier")
    st.write("Upload a PDF of a medical leaflet to simplify its content.")

    api_key = st.text_input("Enter your Anthropic API Key", type="password")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if api_key and uploaded_file is not None:
        if st.button("Process PDF"):
            try:
                st.write("Processing your file...")
                original_text = extract_text_from_pdf(uploaded_file)

                simplified_text = simplify_text_with_claude(original_text, api_key)
                
                if simplified_text.startswith("Error:"):
                    st.error(simplified_text)
                else:
                    metrics = analyze_text(simplified_text)

                    st.subheader("Original vs Simplified Text")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Text**")
                        st.text_area("", value=original_text, height=400, disabled=True)
                    with col2:
                        st.markdown("**Simplified Text**")
                        st.text_area("", value=simplified_text, height=400, disabled=True)

                    st.subheader("Readability Metrics")
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value:.2f}")

                    iteration = 1
                    max_iterations = 5

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while not is_text_readable(metrics) and iteration < max_iterations:
                        status_text.text(f"Iteration {iteration}: Simplifying further...")
                        simplified_text = simplify_text_with_claude(simplified_text, api_key, metrics)
                        metrics = analyze_text(simplified_text)
                        iteration += 1
                        progress_bar.progress(iteration / max_iterations)

                    progress_bar.progress(100)
                    
                    if is_text_readable(metrics):
                        status_text.text("The simplified text meets the readability criteria.")
                    else:
                        status_text.text(f"The simplified text still doesn't meet all readability criteria after {max_iterations} iterations.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter your Anthropic API Key and upload a PDF file to proceed.")

if __name__ == "__main__":
    main()
