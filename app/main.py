import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile,File
from app.utils import pdf_to_text,text_to_vector
from app.features.relevence_score import relevence_score_function

app = FastAPI()

@app.get("/")
def read_root():
    return {"Welcome Career-AI":"V1"}

@app.get("/health-check")
def health():
    return {"condition":"healthy"}


@app.post("/extract-text-from-pdf")
async def text_extract(file: UploadFile = File(...)):
    text=""
    text=pdf_to_text(file)
    return text

@app.post("/text-to-vectors")
async def texttovector(textfile: UploadFile=File(...)):
    content_bytes = await textfile.read()
    text_content = content_bytes.decode("utf-8")
    vectors = text_to_vector(text_content)
    
    return vectors

@app.post("/get-relevence-score")
async def relevencescore(resume: UploadFile=File(...),JD: UploadFile=File(...)):
    
    resume_text=pdf_to_text(resume)
    jd_text=pdf_to_text(JD)
    
    resume_vectors= await text_to_vector(resume_text)
    jd_vectors= await text_to_vector(jd_text)
    
    score=relevence_score_function(resume_vectors,jd_vectors)
    
    
    return {
        "Resume and JD relevence score is": f"{int(score*100)}",
        "message":"Success"
        }
    


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)