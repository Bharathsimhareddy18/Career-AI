import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile,File,Form
from app.utils import pdf_to_text,text_to_vector,LLM_distilliation_for_resume,LLM_distilliation_for_jd,LLM_distilliation_rich_user_data, career_roadmap_gen,fetch_leetcdoe_userdata
from app.features.relevence_score import relevence_score_function
import asyncio

app = FastAPI()

@app.get("/")
def read_root():
    return {"Welcome Career-AI":"V1",
            "Status":"Healthy"
            }

@app.post("/get-relevence-score")
async def relevencescore(resume: UploadFile=File(...),JD: UploadFile=File(...)):
    
    resume_text=pdf_to_text(resume)
    jd_text=pdf_to_text(JD)
    
    
    
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
    
    
    return {
        "Resume and JD relevence score is": f"{int(score*100)}",
        "message":"Success"
        }
    
    
@app.post("/Career-roadmap")
async def careerroadmap(resume: UploadFile=File(...), Jobrole : str = Form(...), hours : float =Form(...)):
    
    resume_content=pdf_to_text(resume)
    resume_distilled=await LLM_distilliation_rich_user_data(resume_content)
    
    if not resume_distilled.is_valid_document:
        return {"Error":"Uploaded file is not a valid resume"}
    
    roadmap=await career_roadmap_gen(resume_distilled,Jobrole,hours)
    
    return {
        "User profile":resume_distilled.model_dump(),
        "result":roadmap.model_dump()
        }

@app.post("/DSA-roadmap")
async def dsa_roadmap_gen(leetcode_url:str):
    
    leetcode_url = leetcode_url.replace("https://", "").replace("http://", "").replace("www.", "")
    
    if leetcode_url.startswith("leetcode.com/u/"):
        
        leetcode_url= leetcode_url.replace("leetcode.com/u/","")
        
        username=leetcode_url.split("/")[0].split("?")[0]
        
        
        data = await fetch_leetcdoe_userdata(username)
    
    
    
    
    return {"data":data}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)