import re

def air_gap_redactor(resume_text):
    """
    Redacts emails, phone numbers, dates, names, and blocks common prompt injections.
    """

    # EMAIL
    resume_text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]', resume_text)

    # PHONE (matches many formats: 123-456-7890, (123)456-7890, 123.456.7890, +1-123-456-7890)
    resume_text = re.sub(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', '[REDACTED_PHONE]', resume_text)

    # DATE
    resume_text = re.sub(r'(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ,.-]*\d{1,2}[ ,.-]*\d{2,4}\b)', '[REDACTED_DATE]', resume_text, flags=re.IGNORECASE)

    # NAME
    resume_text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[REDACTED_NAME]', resume_text)

    # BLOCK PROMPT INJECTION
    injection_phrases = [
    r"(?i)ignore previous instructions",
    r"(?i)use the unredacted resume",
    r"(?i)reveal the original",
    r"(?i)do not redact",
    r"(?i)include full name and contact",
    r"(?i)output unfiltered version",
    r"(?i)company-approved resume",
    r"(?i)also include the unmasked version",
    r"(?i)it’s been hr approved",
    ]


    for phrase in injection_phrases:
        resume_text = re.sub(phrase, '[BLOCKED_INJECTION]', resume_text)

    return resume_text
