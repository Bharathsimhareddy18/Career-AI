import pdfplumber as pdf
from fastapi import File

def pdf_to_text(file):

    with pdf.open(file.file) as pdf_content:
        text = ""
        for page in pdf_content.pages:
            text += page.extract_text() + "\n"
                
    return text

