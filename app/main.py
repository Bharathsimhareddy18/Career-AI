import uvicorn
from fastapi import FastAPI
import json
import supabase
from fastapi import UploadFile,File,Form
from app.utils import (
    pdf_to_text,text_to_vector,
    LLM_distilliation_for_resume,
    LLM_distilliation_for_jd,
    LLM_distilliation_rich_user_data, 
    career_roadmap_gen,
    fetch_leetcdoe_userdata,
    suggested_questions,
    DSA_roadmap_gen_llm,
    resume_and_jd_diff,
    docx_to_text
    )
from app.features.relevence_score import relevence_score_function
from fastapi.middleware.cors import CORSMiddleware
from app.output_models import leetcode_user
import asyncio

app = FastAPI()


origins=["http://localhost:3000","http://localhost:8000","http://127.0.0.1.8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CATEGORY_MAP = {}
COMPANY_GROUPS = {"product": [], "service": [], "startup": []}

try:
    with open("./app/company_categories.json", "r") as f:
        CATEGORY_MAP = json.load(f)
        
    for company, category in CATEGORY_MAP.items():
        clean_cat = category.lower()
        if "product" in clean_cat: key = "product"
        elif "service" in clean_cat: key = "service"
        elif "startup" in clean_cat: key = "startup"
        else: key = None
        
        if key:
            COMPANY_GROUPS[key].append(company)
            
    print(f"Loaded {len(CATEGORY_MAP)} companies into categories.")
except Exception as e:
    print(f"Warning: Could not load company_categories.json: {e}")


def parse_file(file_obj: UploadFile, file_bytes: bytes) -> str:
    filename = file_obj.filename.lower()
    
    if filename.endswith(".pdf"):
        return pdf_to_text(file_bytes)
    elif filename.endswith(".docx"):
        return docx_to_text(file_bytes)
    else:
        return " "
    
    
@app.get("/")
def read_root():
    return {"Welcome Career-AI":"V1",
            "Status":"Healthy"
            }

@app.post("/get-relevence-score")
async def relevencescore(resume: UploadFile=File(...),JD: UploadFile=File(...)):
    
    resume_bytes = await resume.read()
    jd_bytes = await JD.read()
    
    resume_text, jd_text = await asyncio.gather(
        asyncio.to_thread(parse_file, resume, resume_bytes),
        asyncio.to_thread(parse_file, JD, jd_bytes)
    )
    
    if not resume_text.strip():
        return {"Error": "Resume format unsupported or empty. Use PDF or DOCX."}
    if not jd_text.strip():
        return {"Error": "JD format unsupported or empty. Use PDF or DOCX."}
    
    resume_distilled,jd_distilled = await asyncio.gather( 
    LLM_distilliation_for_resume(resume_text),
    LLM_distilliation_for_jd(jd_text)
    )
    
    if not resume_distilled.is_valid_document:
        return{"Error":"Uploaded file is not a valid resume."}
    if not jd_distilled.is_valid_document:
        return{"Error":"Uploaded file is not a valid job resume."}
    
    resume_str=f"Role:{resume_distilled.role}|Skills:{resume_distilled.skills}"
    jd_str=f"Role:{jd_distilled.role}|Skills:{jd_distilled.skills}"
    
    resume_vectors,jd_vectors= await asyncio.gather(
    text_to_vector(resume_str),
    text_to_vector(jd_str))
    
    score=relevence_score_function(resume_vectors,jd_vectors)
    
    result=await resume_and_jd_diff(resume_str,jd_str,score)
    
    
    
    
    return {
        "Resume and JD relevence score is": f"{int(score*100)}",
        "Difference":result,
        "message":"Success"
        }
    
    
@app.post("/Career-roadmap")
async def careerroadmap(resume: UploadFile=File(...), Jobrole : str = Form(...), hours : float =Form(...)):
    
    resume_bytes = await resume.read()
    
    resume_content=await asyncio.to_thread(parse_file, resume, resume_bytes)
    
    if not resume_content.strip():
        return {"Error": "Resume format unsupported or empty. Use PDF or DOCX."}
   
    
    resume_distilled=await LLM_distilliation_rich_user_data(resume_content)
    
    if not resume_distilled.is_valid_document:
        return {"Error":"Uploaded file is not a valid resume"}
    
    roadmap=await career_roadmap_gen(resume_distilled,Jobrole,hours)
    
    return {
        "User profile":resume_distilled.model_dump(),
        "result":roadmap.model_dump()
        }


@app.get("/DsaConfig")
def DSAconfig():
    return CATEGORY_MAP

@app.get("/Dsalist")
async def DSAconfig(target_company,):
    questions=await suggested_questions(target_company,CATEGORY_MAP,COMPANY_GROUPS)
    return questions


@app.post("/DSA-roadmap")
async def dsa_roadmap_gen(leetcode_public:str,user_target_company:str,time_period_for_interview:int):
    
    leetcode_url =leetcode_public.replace("https://", "").replace("http://", "").replace("www.", "")
    
    if leetcode_url.startswith("leetcode.com/u/"):
        
        leetcode_url= leetcode_url.replace("leetcode.com/u/","")
        
        username=leetcode_url.split("/")[0].split("?")[0]
        
        
        data,recommended_list= await asyncio.gather(
        fetch_leetcdoe_userdata(username),
        suggested_questions(user_target_company,CATEGORY_MAP,COMPANY_GROUPS))
       
    roadmap=await DSA_roadmap_gen_llm(data,user_target_company,recommended_list,time_period_for_interview)
    
    
    
    return {
        "User_data":data,
        "Roadmap":roadmap
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)