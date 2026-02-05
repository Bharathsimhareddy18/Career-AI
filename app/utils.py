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


async def LLM_distilliation_for_resume(text):
    result=""
     
    SYSTEM_PROMPT = """
You are an expert Technical Recruiter. Your goal is to extract a precise list of hard technical skills and the target job role.

Rules:
1. Extract skills from ANYWHERE in the text (Skills section, Projects, Experience).
2. Normalize skills (e.g., "React.js" -> "React", "AWS EC2" -> "AWS").
3. IGNORE soft skills (Leadership, Communication).
4. IGNORE general terms (Coding, Programming).
5. Output format: "Role: <Predicted Role> | Skills: <Skill1>, <Skill2>, ..."

Example Output:
Role: Backend Engineer | Skills: Python, FastAPI, Docker, PostgreSQL, AWS, Redis
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
            }]
    )    
    
    result=response.choices[0].message.content
    
    return result


async def LLM_distilliation_for_jd(text):
    result=""
     
    SYSTEM_PROMPT = """
You are an expert Technical Recruiter. Your goal is to extract a precise list of hard technical skills and the target job role.

Rules:
1. Extract skills from ANYWHERE in the text (Skills section, Projects, Experience).
2. Normalize skills (e.g., "React.js" -> "React", "AWS EC2" -> "AWS").
3. IGNORE soft skills (Leadership, Communication).
4. IGNORE general terms (Coding, Programming).
5. Output format: "Role: <Predicted Role> | Skills: <Skill1>, <Skill2>, ..."

Example Output:
Role: Backend Engineer | Skills: Python, FastAPI, Docker, PostgreSQL, AWS, Redis
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
            }]
    )    
    
    result=response.choices[0].message.content
    
    return result


