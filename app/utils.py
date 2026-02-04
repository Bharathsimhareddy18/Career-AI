import pdfplumber as pdf
from fastapi import File
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


def pdf_to_text(file):

    with pdf.open(file.file) as pdf_content:
        text = ""
        for page in pdf_content.pages:
            text += page.extract_text() + "\n"
                
    return text

async def text_to_vector(text):
    model_name = "text-embedding-3-small"
    client=AsyncOpenAI()
    
    response=await client.embeddings.create(
        input=text,
        model=model_name
    )
    vectors=response.data[0].embedding
    
    return vectors
