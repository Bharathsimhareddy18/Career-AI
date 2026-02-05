# Career AI

**An AI-powered Career Coach that bridges the gap between your resume and your dream job.**

Career AI is an intelligent backend system designed to analyze resumes, compare them against job descriptions (JDs), and generate personalized career roadmaps. It uses Large Language Models (LLMs) and Vector Embeddings to provide mathematically precise relevance scores and actionable insights.

![Career AI Output](https://github.com/user-attachments/assets/383ede50-4309-46d2-9c60-ce4ae2a7aac7)
*Current Output: Resume vs. JD Relevance Score using Cosine Similarity*

---

## Features

-   **Resume Distiller (Live):**
    -   Ingests PDF Resumes and Job Descriptions.
    -   Extracts semantic meaning using LLMs.
    -   Calculates a **Relevance Score (0-100%)** using Cosine Similarity on vector embeddings.
-   **Career Roadmap Generator (In Progress):**
    -   Generates personalized learning paths based on skill gaps.
    -   Suggests projects and resources.
-   **LeetCode Analyzer (Planned):**
    -   Tailors DSA preparation based on target companies.

---

## Tech Stack

-   **Framework:** FastAPI (Python) - Fully Async/Await
-   **AI/LLM:** OpenAI GPT-4 / Gemini
-   **Vector Math:** Scikit-learn (Cosine Similarity)
-   **PDF Processing:** PyPDF2 / Custom Extractors

---

## Getting Started

### Prerequisites
-   Python 3.9+
-   OpenAI/Gemini API Key

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Bharathsimhareddy18/Career-AI.git
    cd Career-AI
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
    Create a `.env` file and add your API keys:
    ```env
    OPENAI_API_KEY=your_key_here
    ```

5.  **Run the Server**
    ```bash
    uvicorn main:app --reload
    ```

6.  **Access the API**
    Open your browser and navigate to the Swagger UI:
    `http://127.0.0.1:8000/docs`

---

## API Endpoints

### `POST /get-relevance-score`
Uploads a Resume (PDF) and a Job Description (PDF) to calculate how well they match.


**Response:**
```json
{
  "Resume and JD relevance score is": "96",
  "message": "Success"
}
