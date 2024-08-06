
import streamlit as st
import PyPDF2
import google.generativeai as genai
import re
import math
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
    model = genai.GenerativeModel("gemini-1.5-pro-exp-0801")
    
    base_prompt = """
You are an AI assistant specialized in simplifying pharmaceutical and medical instructions. Your task is to rewrite the given text to be easily understood by people with limited health literacy, aiming for a 12-year-old reading level. Follow these guidelines:

1. Maintain all legal and safety information, including specific instructions for special groups and overdose situations.
2. Keep the text length similar to the original.
3. Reduce long sentences, nested clauses, passive voice, nominalizations, long words, multi-syllable words, abstract nouns, and medical jargon.
4. Include information on use during pregnancy and breastfeeding, and measures for overdose.
5. Use the following format for the simplified text:
   a. Was ist {Produkt} und wofür wird es angewendet?
   b. Was sollten Sie vor der <Einnahme> <Anwendung> von {Produkt} beachten?
   c. Wie ist {Produkt} <einzunehmen> <anzuwenden>?
   d. Welche Nebenwirkungen sind möglich?
   e. Wie ist {Produkt} aufzubewahren?

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

Now, please simplify the following text:

{text_to_simplify}

Ensure your simplified version maintains all important information while being more accessible to readers with limited health literacy.
    """
    
    if metrics:
        metric_feedback = f"""
Current readability metrics:
- Flesch Reading Ease: {metrics['Flesch Reading Ease']:.2f} (target: 60-70)
- Flesch-Kincaid Grade: {metrics['Flesch-Kincaid Grade']:.2f} (target: 6-8)
- Gunning Fog: {metrics['Gunning Fog']:.2f} (target: 8-10)
- SMOG Index: {metrics['SMOG Index']:.2f} (target: 7-9)
- Coleman-Liau Index: {metrics['Coleman-Liau Index']:.2f} (target: 7-9)
- Automated Readability Index: {metrics['Automated Readability Index']:.2f} (target: 7-9)

Please adjust the text to improve these metrics while maintaining accuracy and completeness.
        """
        base_prompt += f"\n\n{metric_feedback}"

    full_prompt = base_prompt.replace("{text_to_simplify}", text)
    
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
            for metric in ['Flesch Reading Ease', 'Flesch-Kincaid Grade', 'Gunning Fog', 'SMOG Index', 
                           'Coleman-Liau Index', 'Automated Readability Index']:
                change = current_metrics[metric] - original_metrics[metric]
                improvement = "improved" if change > 0 else "decreased"
                st.write(f"{metric} {improvement} by {abs(change):.2f} points")
                               
            with st.expander("Additional Statistics"):
                st.write(f"Number of sentences: {metrics['Number of sentences']:.0f}")
                st.write(f"Number of words: {metrics['Number of words']:.0f}")
                st.write(f"Average words per sentence: {metrics['Average words per sentence']:.2f}")
    else:
        st.warning("Please enter your Gemini API Key and upload a PDF file to proceed.")

if __name__ == "__main__":
    main()
