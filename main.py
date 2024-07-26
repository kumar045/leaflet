import streamlit as st
import PyPDF2
import google.generativeai as genai
import textstat

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
    model = genai.GenerativeModel('gemini-pro')
    
    base_prompt = """
    Simplify the following medical leaflet text for a general audience:
    1. Use simple, clear language suitable for ages 12-14 (around 8th grade).
    2. Break down complex medical terms and explain them in plain language.
    3. Use short sentences and paragraphs.
    4. Organize the information with clear headings and bullet points where appropriate.
    5. Prioritize the most important information that patients need to know.
    6. Maintain all critical safety information, but phrase it in a more accessible way.
    7. Aim for a total length of about 500-700 words.
    """
    
    if metrics:
        metric_feedback = """
        Current readability metrics:
        - Flesch Reading Ease: {fre:.2f} (target: 60-70)
        - Flesch-Kincaid Grade: {fkg:.2f} (target: 6-8)
        - Gunning Fog: {gf:.2f} (target: 8-10)
        - SMOG Index: {smog:.2f} (target: 7-9)
        - Coleman-Liau Index: {cli:.2f} (target: 7-9)
        - Automated Readability Index: {ari:.2f} (target: 7-9)
        
        Please adjust the text to improve these metrics while maintaining accuracy and completeness.
        """
        base_prompt += metric_feedback.format(
            fre=metrics["Flesch Reading Ease"],
            fkg=metrics["Flesch-Kincaid Grade"],
            gf=metrics["Gunning Fog"],
            smog=metrics["SMOG Index"],
            cli=metrics["Coleman-Liau Index"],
            ari=metrics["Automated Readability Index"]
        )
    
    full_prompt = base_prompt + "\n\nHere's the text to simplify:\n\n" + text
    response = model.generate_content(full_prompt)
    return response.text

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

    api_key = st.text_input("Enter your Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if api_key and uploaded_file is not None:
        if st.button("Process PDF"):
            st.write("Processing your file...")
            original_text = extract_text_from_pdf(uploaded_file)

            with st.expander("Original Text"):
                st.write(original_text)

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

            st.subheader("Simplified Text")
            st.write(simplified_text)

            st.subheader("Readability Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Flesch Reading Ease", f"{metrics['Flesch Reading Ease']:.2f}")
                st.metric("Flesch-Kincaid Grade", f"{metrics['Flesch-Kincaid Grade']:.2f}")
                st.metric("Gunning Fog", f"{metrics['Gunning Fog']:.2f}")
            with col2:
                st.metric("SMOG Index", f"{metrics['SMOG Index']:.2f}")
                st.metric("Coleman-Liau Index", f"{metrics['Coleman-Liau Index']:.2f}")
                st.metric("Automated Readability Index", f"{metrics['Automated Readability Index']:.2f}")

            with st.expander("Additional Statistics"):
                st.write(f"Number of sentences: {metrics['Number of sentences']:.0f}")
                st.write(f"Number of words: {metrics['Number of words']:.0f}")
                st.write(f"Number of complex words: {metrics['Number of complex words']:.0f}")
                st.write(f"Percentage of complex words: {metrics['Percentage of complex words']:.2f}%")
                st.write(f"Average words per sentence: {metrics['Average words per sentence']:.2f}")
                st.write(f"Average syllables per word: {metrics['Average syllables per word']:.2f}")
    else:
        st.warning("Please enter your Gemini API Key and upload a PDF file to proceed.")

if __name__ == "__main__":
    main()
