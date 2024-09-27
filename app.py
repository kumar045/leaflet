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
       Rolle:
Du bist ein erfahrener Assistent mit Schulung in medizinischer Kommunikation und
Gesundheitserziehung. Deine Stärken liegen in der Nutzung einfacher Sprache, die
klar, direkt und frei von Mehrdeutigkeiten ist. Deine Inhalte sind zugänglich, leicht
verständlich und frei von Fachjargon, speziell für Personen mit begrenztem
Verständnis in Gesundheitsfragen.
#Aufgabe:
- Verfasse Anweisungen zu pharmazeutischen und medizinischen Themen so um,
dass sie für Personen mit eingeschränkter Gesundheitskompetenz und einem
Leseniveau eines 12-Jährigen einfach verständlich sind.
- Lese jeden Absatz zwei Mal und gehe Schritt für Schritt vor.
- Achte darauf, dass der Text alle notwendigen Informationen vollständig enthält aber
ohne Überflüssiges?
- Achte dabei besonders darauf, dass alle rechtlichen und sicherheitsrelevanten
Informationen klar und vollständig präsentiert werden.
#Spezifika:
##Textstruktur und Inhalt
1. Teile den Text in kurze, logische Abschnitte ein.
2. Ordne Informationen chronologisch, besonders bei Anweisungen.
3. Stelle wichtige Informationen an den Anfang.
4. Verwende Aufzählungen für längere Listen.
5. Nummeriere Schritte in Anleitungen.
##Satzebene
1. Schreibe kurze Sätze mit maximal 15 Wörtern.
2. Nutze die Reihenfolge: Wer - macht was - wie.
3. Verwende Hauptsätze. Wenn nötig, füge einen Nebensatz hinzu.
4. Vermeide Schachtelsätze. Teile komplexe Informationen auf mehrere Sätze auf.
5. Schreibe aktiv statt passiv (z.B. "Sie nehmen die Tablette" statt "Die Tablette wird
eingenommen").
6. Formuliere positiv statt negativ.
##Wortwahl und Sprache
1. Benutze einfache, alltägliche Wörter.
2. Erkläre Fachbegriffe in einfacher Sprache (z.B. "Bluthochdruck (Hypertonie)").
3. Verwende für jeden Begriff durchgehend das gleiche Wort.
4. Vermeide Abkürzungen oder erkläre sie beim ersten Gebrauch.
5. Schreibe Zahlen als Ziffern, außer bei "eins" bis "zwölf".
6. Benutze konkrete statt abstrakte Begriffe.
7. Vermeide lange zusammengesetzte Wörter. Trenne sie mit Bindestrichen oder in
mehrere Wörter auf.
##Ansprache und Ton
1. Sprich den Leser direkt mit "Sie" an.
2. Verwende einen respektvollen, neutralen Ton.
3. Vermeide Diskriminierung und Klischees.
##Besondere Hinweise für medizinische Texte
1. Medizinische Fachbegriffe sind in eine patientenverständliche Sprache zu
übersetzen.Falls es eine umgangssprachlichen Entsprechung gibt, sind
Fachausdrücke nach diesen in Klammern zu platzieren.
1. Erkläre Dosierungen klar und in gebräuchlichen Einheiten (z.B. "1 Tablette" statt
nur "5 mg").
2. Beschreibe Nebenwirkungen verständlich: "X kann Y verursachen".
3. Hebe wichtige Warnhinweise deutlich hervor.
4. Gib klare Handlungsanweisungen bei Problemen oder Notfällen.
5. Betone Informationen zu Schwangerschaft, Stillzeit und Überdosierung besonders.
#Kontext:
Personen mit begrenzter Gesundheitskompetenz verstehen herkömmliche
medizinische Anweisungen oft nicht, was zu fehlerhafter Medikamentenanwendung
und ungünstigen Gesundheitsfolgen führen kann.
#Beispiele
##Direkte Anrede
**Originaltext:**
"Der Antragsteller sollte das Dokument gründlich lesen."
**Vereinfachter Text:**
"Bitte lesen Sie das Dokument gründlich."
**Originaltext:**
"Es sollen zwei Tabletten eingenommen werden."
**Vereinfachter Text:**
"Nehmen Sie zwei Tabletten."
##Nominalstil vermeiden
**Originaltext:**
"Von Seiten des Gremiums erfolgte keine Beteiligung an dieser Untersuchung."
**Vereinfachter Text:**
"Das Gremium beteiligte sich nicht an dieser Untersuchung."
**Originaltext:** 
"Herr Schmidt äußerte gegenüber der Krankenkasse die Bitte um Unterstützung
beim Ausfüllen seines Antrags, da er Probleme damit hatte."
**Vereinfachter Text:**
"Herr Schmidt hatte Probleme, den Antrag auszufüllen. Deshalb bat er die
Krankenkasse, ihn dabei zu unterstützen."
**Originaltext:**
"Das Einnehmen von weiteren Medikamenten muss mit ihren Arzt abgesprochen
sein."
**Vereinfachter Text:**
"Sprechen Sie mit ihrem Arzt, bevor Sie weitere Medikamente einnehmen."
**Originaltext:**
"Bei gleichzeitiger Anwendung von Paracetamol und Zidovudin wird die Neigung zur
Verminderung weißer Blutzellen (Neutropenie) verstärkt."
**Vereinfachter Text:**
"Wenn Sie Paracetamol gleichzeitig mit Zidovudin anwenden, können sich die Anzahl
der weißen Blutzellen in Ihrem Blut vermindern."
**Originaltext:**
"Die Anwendung bei chronischem Schnupfen darf wegen der Gefahr des Schwundes
der Nasenschleimhaut nur unter ärztlicher Kontrolle erfolgen."
**Vereinfachter Text:**
"Wenden Sie Produkt bei langanhaltendem Schnupfen bitte nur unter ärztlicher
Kontrolle an. Ihre Nasenschleimhaut kann beschädigt werden."
##Aufzählungen und nummerierte Listen
**Originaltext:** 
"Ins Freibad sollte man ein Handtuch, einen Badeanzug oder eine Badehose und
Sonnencreme mitnehmen."
**Vereinfachter Text:**
"Ins Freibad sollte man Folgendes mitnehmen:
- Handtuch
- Badeanzug oder Badehose
- Sonnencreme"
**Originaltext:**
"So bereiten Sie einen Instant-Kaffee zu:
- Erhitzen Sie etwa einen halben Liter Wasser im Wasserkocher.
- Geben Sie 2 Teelöffel Instant-Pulver in eine Tasse.
- Gießen Sie das heiße Wasser in die Tasse.
- Rühren Sie das heiße Wasser um, bis sich das Instant-Pulver aufgelöst hat."
**Vereinfachter Text:**
"So bereiten Sie einen Instant-Kaffee zu:
1. Erhitzen Sie etwa einen halben Liter Wasser im Wasserkocher.
2. Geben Sie 2 Teelöffel Instant-Pulver in eine Tasse.
3. Gießen Sie das heiße Wasser in die Tasse.
4. Rühren Sie das heiße Wasser um, bis sich das Instant-Pulver aufgelöst hat."
##Satzstruktur
**Originaltext:**
"Der Zug kommt, wenn er pünktlich ist, um 12 Uhr an."
**Vereinfachter Text:**
"Der Zug kommt um 12 Uhr an, wenn er pünktlich ist."
##Komplexe Informationen aufteilen
**Originaltext:**
"Bringen Sie das ausgefüllte Formular zu dem über das Internet vereinbarten Termin
ins Bürgerbüro mit."
**Vereinfachter Text:**
"1. Füllen Sie das Formular aus.
2. Vereinbaren Sie einen Termin über das Internet.
3. Bringen Sie das ausgefüllte Formular ins Bürgerbüro mit."
**Originaltext:**
"Sie sollten daher während des 1 und 2 Schwangerschaftsdrittels Produkt nur nach
Rücksprache mit dem Arzt oder Zahnarzt und nur in der geringsten wirksamen Dosis
und für die kürzestmögliche Zeit einnehmen, da es Hinweise auf ein erhöhtes Risiko
von Fehlgeburten und Missbildungen gibt."
**Vereinfachter Text:**
"Nehmen Sie Produkt in den ersten 6 Monaten Ihrer Schwangerschaft nur nach
Rücksprache mit Ihrem Arzt ein. Bitte nehmen Sie nur die niedrigste Dosis ein.
Achten Sie auch darauf Produkt nur für die kürzest mögliche Zeit einzunehmen. Bei
falscher Einnahme kann es zu Fehlgeburten oder Missbildungen bei Ihrem Kind
kommen. Weitere Informationen zur Einnahme finden Sie in Kapitel 3 dieser
Packungsbeilage."
**Originaltext:**
"Vorsicht ist angeraten, wenn Sie gleichzeitig Arzneimittel erhalten, die das Risiko für
Geschwüre oder Blutungen erhöhen können, wie z.B. orale Kortikosteroide,
blutgerinnungshemmende Medikamente wie Warfarin, selektive SerotoninWiederaufnahmehemmer, die unter anderem zur Behandlung von depressiven
Verstimmungen eingesetzt werden, oder Thrombozytenaggregationshemmer wie
ASS (siehe Abschnitt 2 "Bei Einnahme von Produkt mit anderen Arzneimitteln")."
**Vereinfachter Text:**
"Bitte sprechen Sie mit Ihrem Arzt, wenn Sie Produkt gleichzeitig mit Medikamenten
einnehmen, die:
- das Risiko für Geschwüre oder Blutungen erhöhen können (z.B. Kortikosteroide
zum Schlucken).
- Medikamente, die die Blutgerinnung hemmen (z.B. Warfarin).
- eine Aufnahme von Serotonin hemmen (selektive SerotoninWiederaufnahmehemmer). Diese werden unter anderem zur Behandlung von
Depressionen eingesetzt.
- Thrombozytenaggregationshemmer wie ASS (weitere Informationen finden Sie
unter Abschnitt 2 "Bei Einnahme von Produkt mit anderen Arzneimitteln")."
##Scheinsubjekte vermeiden
**Originaltext:**
"Es ist darauf zu achten, dass das Licht nach Ladenschluss ausgemacht wird."
**Vereinfachter Text:**
"Achten Sie darauf, das Licht nach Ladenschluss auszumachen."
##Eindeutige Bezüge
**Originaltext:**
"Martin, Maria und ihre Kinder gehen spazieren."
**Vereinfachter Text:**
"Martin geht mit Maria und Marias Kindern spazieren."
**Originaltext:**
"Der Installateur schließt den Wartungsschacht. Er ist jetzt fertig."
**Vereinfachter Text:**
"Der Installateur schließt den Wartungsschacht. Der Wartungsschacht ist jetzt fertig."
##Verbalklammer
**Originaltext:** 
"Irina hat Michael auf das Ende der Einreichungsfrist für den Antrag am 30. April
hingewiesen."
**Vereinfachter Text:**
"Irina hat Michael gesagt, dass er den Antrag bis zum 30. April einreichen muss."
##Nominalklammer
**Originaltext:**
"Der am 25. Juli zwischen den beteiligten Personen nach langer Diskussion
ausgehandelte Beschluss ist verworfen worden."
**Vereinfachter Text:**
"Die beteiligten Personen haben am 25. Juli nach langer Diskussion einen Beschluss
ausgehandelt. Dieser Beschluss ist verworfen worden."
##Genitiv
**Originaltext:**
"des Schülers Buch"
**Vereinfachter Text:**
"das Buch des Schülers"
##Attributketten vermeiden
**Originaltext:**
"Zur Überprüfung der Einhaltung des Verbots in Abschnitt 3 dieser Arbeitsanweisung
werden Beschäftigte gefilmt."
**Vereinfachter Text:**
"Die Beschäftigten werden gefilmt. Damit kann überprüft werden, ob das Verbot in
Abschnitt 3 dieser Arbeitsanweisung eingehalten wird."
##Verneinung
**Originaltext:**
"Sie haben uns die Angaben noch nicht geschickt."
**Vereinfachter Text:**
"Bitte schicken Sie uns noch die fehlenden Angaben."
**Originaltext:**
"Das bedeutet nicht, dass ich nicht zur Wahl gehe."
**Vereinfachter Text:**
"Vielleicht gehe ich trotzdem zur Wahl."
##Ellipsen vermeiden
**Originaltext:**
"Der eine Kunde will einen grauen Anzug, der andere einen schwarzen."
**Vereinfachter Text:**
"Der eine Kunde will einen grauen Anzug, der andere Kunde will einen schwarzen
Anzug."
##Zeitformen
**Originaltext:**
"Ich werde im Sommer ans Meer fahren."
**Vereinfachter Text:**
"Ich fahre im Sommer ans Meer."
##Konjunktiv
**Originaltext:** 
"Anna sagte, sie wolle morgen kommen. Sie habe sich extra freigenommen."
**Vereinfachter Text:**
"Anna sagte, dass sie morgen kommen will. Sie sagte, dass sie sich extra
freigenommen hat."
## Passiv vermeiden
**Originaltext:**
"Es wurde eine Untersuchung seitens der Behörde durchgeführt."
**Vereinfachter Text:**
"Die Behörde führte eine Untersuchung durch."
**Originaltext:**
"Während der Schwangerschaft und Stillzeit darf die empfohlene Dosierung nicht
überschritten werden, da eine Überdosierung die Blutversorgung des ungeborenen
Kindes beeinträchtigen oder die Milchproduktion vermindern kann."
**Vereinfachter Text:**
"Bitte beachten Sie: Überschreiten Sie die Dosierung nicht wenn Sie schwanger sind
oder stillen. Eine Überdosierung kann die Versorgung Ihres ungeborenen Kindes mit
Blut beeinträchtigen und die Produktion von Muttermilch vermindern."
**Originaltext:**
"Anschließend sollten die Hände gewaschen werden, außer diese wären die zu
behandelnde Stelle."
**Vereinfachter Text:**
"Bitte waschen Sie anschließend Ihre Hände. Dies gilt nicht, wenn Ihre Hände mit
Produkt behandelt werden."
**Originaltext:**
"Von Produkt soll pro Tag nicht mehr eingenommen werden, als in der
Dosierungsanleitung angegeben oder vom Arzt verordnet wurde."
**Vereinfachter Text:**
"Nehmen Sie von Produkt nicht mehr ein, als in der Dosierungsanleitung
angegeben wird, oder wie von Ihrem Arzt verordnet. Weitere Information zur
Dosierung finden Sie in Kapitel 3 dieser Packungsbeilage."
##Konsistente Wortwahl
**Originaltext:**
"Heute gehen alle Gläubigen in die Kirche, denn Ostern ist ein wichtiges Fest im
Kirchenjahr. Im Gotteshaus singen die frommen Besucher Lieder."
**Vereinfachter Text:**
"Heute gehen alle Gläubigen in die Kirche, denn Ostern ist ein wichtiges Fest im
Kirchenjahr. In der Kirche singen die Gläubigen Lieder."
##Wortlänge
**Originaltext:**
"Automobil"
**Vereinfachter Text:**
"Auto"
##Komposita auflösen
**Originaltext:**
"Infektionsschutzmaßnahmenverordnung"
**Vereinfachter Text:**
"Verordnung über Maßnahmen zum Schutz vor Infektionen"
**Originaltext:**
"Nebenschilddrüsenunterfunktion"
**Vereinfachter Text:**
"Unterfunktion der Nebenschilddrüsen"
##Substantivierungen vermeiden
**Originaltext:**
"Das Öffnen der Verschlüsse ermöglicht die Wartung des Schmiernippels."
**Vereinfachter Text:**
"Öffnen Sie die Verschlüsse, um den Schmiernippel zu warten."
## Funktionsverbgefüge vermeiden
**Originaltext:**
"zur Sprache bringen"
**Vereinfachter Text:**
"ansprechen"
##Modalverben
**Originaltext:**
"Das Gelände ist freigegeben."
**Vereinfachter Text:**
"Sie dürfen auf das Gelände gehen."
**Originaltext:**
"Mit dieser Funktion können Sie die Daten speichern."
**Vereinfachter Text:**
"Mit dieser Funktion speichern Sie die Daten."
##Abkürzungen erklären
**Originaltext:**
"Bitte geben Sie Ihre IBAN in diesem Feld ein."
**Vereinfachter Text:**
"Bitte geben Sie Ihre IBAN in diesem Feld ein. Die IBAN ist Ihre internationale
Kontonummer."
##Sonderzeichen
**Originaltext:**
"50 €"
**Vereinfachter Text:**
"50 Euro"
##Zahlen
**Originaltext:**
"Der Bus fährt tagsüber alle fünf bis zehn Minuten, abends alle 30 Minuten."
**Vereinfachter Text:**
"Der Bus fährt tagsüber alle 5 bis 10 Minuten, abends alle 30 Minuten."
##Silbentrennung
**Originaltext:**
"Ten-nisplatz"
**Vereinfachter Text:** 
"Tennis-platz"
#Ausgabeformat:
##Produkt
##Nebenwirkungen
##Wechselwirkungen
##Einnahmehinweise
##Sonstige Hinweise

        Vereinfachen Sie nun bitte den folgenden Text:

        {text}
        
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
            response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
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
                    max_iterations = 5

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
