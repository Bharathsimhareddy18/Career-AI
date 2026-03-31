# Career AI — Intelligent Resume & JD Matching System
An AI-powered backend that analyzes resumes, scores them against job descriptions,
and generates personalized career roadmaps — using LLMs and vector embeddings.

Live: https://career-ai-flax.vercel.app/

---

<img width="1918" height="1006" alt="image" src="https://github.com/user-attachments/assets/20ea90f3-3ab0-4af9-a555-55bf267ac585" />

---

## What It Does

Career AI runs three specialized modules to accelerate your job search:

**Resume Distiller** — Ingests PDF resumes and job descriptions. Extracts semantic
meaning using LLMs and calculates a **Relevance Score (0–100%)** via Cosine Similarity
on vector embeddings. Gives you a mathematically precise match signal instantly.

**Career Roadmap Generator** *(In Progress)* — Analyzes skill gaps between your resume
and the target JD. Generates a personalized learning path with project suggestions
and curated resources to close the gap fast.

**LeetCode Analyzer** *(Planned)* — Tailors DSA preparation based on your target
companies. Tells you exactly what to grind instead of grinding everything.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Fully Async/Await) |
| AI | GPT-4 / Gemini, LLM Tool Calling |
| Vector Math | Scikit-learn (Cosine Similarity) |
| PDF Processing | PyPDF2 / Custom Extractors |
| Frontend | Vercel |

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /get-relevance-score` | Upload Resume + JD PDFs, returns match score (0–100%) |

---

## Screenshots

<img width="1918" height="1006" alt="image" src="https://github.com/user-attachments/assets/20ea90f3-3ab0-4af9-a555-55bf267ac585" />

<img width="1920" height="1004" alt="image" src="https://github.com/user-attachments/assets/35c138d1-2e26-4054-8c74-7f83086aefd6" />

<img width="1920" height="1004" alt="image" src="https://github.com/user-attachments/assets/422be405-96c5-4299-831e-e14c39ac1b5d" />

