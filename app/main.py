import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile,File
from app.utils import pdf_to_text

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



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)