import pdfplumber as pdf
from fastapi import File
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.output_models import jobData
import json

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


async def LLM_distilliation_for_resume(text : str , doc_type: str = "Resume")->jobData:
    result=""
     
    SYSTEM_PROMPT = f"""
    You are an expert Technical Recruiter. Analyze this {doc_type}.
    Extract the 'role' and a list of 'skills'.
    
    CRITICAL RULES:
    1. If the text contains ANY keywords like 'Experience', 'Education', 'Skills', or job titles, assume it IS a valid document and set "is_valid_document": true.
    2. Only set "false" if the text is completely gibberish or unrelated (like a cooking recipe).
    3. Normalize skills (e.g., "React.js" -> "React").
    4. Output must be valid JSON matching this schema:
       {{
         "role": "extracted role or null",
         "skills": ["skill1", "skill2"],
         "is_valid_document": true/false
       }}
    """
     
    client=AsyncOpenAI()
    
    response=await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
                "role": "system", 
                "content": f"{SYSTEM_PROMPT}"
            
            },
            {
                "role": "user", 
                "content": f"Extract key info:\n\n{text[:4000]}" 
            }],
        response_format={"type":"json_object"}
    )    
    
    result=json.loads(response.choices[0].message.content)
    
    return jobData(**result)


async def LLM_distilliation_for_jd(text:str, doc_type : str = "job description")->jobData:
    result=""
     
    SYSTEM_PROMPT = f"""
    You are an expert Technical Recruiter. Analyze this {doc_type}.
    Extract the 'role' and a list of 'skills'.
    
    CRITICAL RULES:
    1. If the text contains ANY keywords like 'Experience', 'Education', 'Skills', or job titles, assume it IS a valid document and set "is_valid_document": true.
    2. Only set "false" if the text is completely gibberish or unrelated (like a cooking recipe).
    3. Normalize skills (e.g., "React.js" -> "React").
    4. Output must be valid JSON matching this schema:
       {{
         "role": "extracted role or null",
         "skills": ["skill1", "skill2"],
         "is_valid_document": true/false
       }}
    """
 
    client=AsyncOpenAI()
    
    response=await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
                "role": "system", 
                "content": f"{SYSTEM_PROMPT}"
            
            },
            {
                "role": "user", 
                "content": f"Extract key info:\n\n{text[:4000]}" 
            }],
        response_format={"type":"json_object"}
    )    
    
    result=json.loads(response.choices[0].message.content)
    
    return jobData(**result)


