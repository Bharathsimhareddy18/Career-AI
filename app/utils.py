import pdfplumber as pdf
from fastapi import File
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.output_models import jobData,UserProfile,CareerRoadmap,GapAnalysis,RoadmapPhase,LeetCodeStats
import requests
import json
import httpx
from supabase._async.client import create_client as create_async_client
import os


model="gpt-5-mini"

#all needed functions here

load_dotenv()

#converts pdf to text
def pdf_to_text(file):

    with pdf.open(file.file) as pdf_content:
        text = ""
        for page in pdf_content.pages:
            text += page.extract_text() + "\n"
                
    return text

#converts into vectors

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
        model=model,
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
    
    
    try:
        
        result=json.loads(response.choices[0].message.content)
    
        return jobData(**result)
    
    except json.JSONDecodeError:
        
        return jobData(
            role = "unknown",
            skills= [],
            is_valid_document = 0,
            summary = None
        )
        
async def resume_and_jd_diff(resume_keys,jd_keys,relevence_score):
    
    client=AsyncOpenAI()
    
    SYSTEM_PROMPT = """
    You are a brutal but efficient Resume Reviewer.
    Analyze the gap between the Resume and JD.
    
    ### RULES FOR OUTPUT:
    1. **Be Punchy:** Max 15 words per bullet point. No long explanations.
    2. **Direct Action:** Start with a verb (e.g., "Add...", "Remove...", "Quantify...").
    3. **Focus on Top 5:** Only list the top 3-5 critical gaps and improvements. Do not list minor nitpicks.

    ### OUTPUT FORMAT (JSON):
    {
      "analysis_summary": "One sentence summary (e.g., 'Good tech stack, but lacks production engineering signals.')",
      "key_gaps": [
        "Missing 'PostgreSQL' and 'CI/CD' keywords.",
        "Role reads 'Intern' but JD expects 'Mid-Level'.",
        "Lack of quantitative metrics (Impact %)."
      ],
      "improvement_actions": [
        "Add 'PostgreSQL', 'Docker', 'Pytest' to Skills section.",
        "Rewrite bullet points to show 'Production' experience.",
        "Add numbers: 'Reduced latency by 20%', 'Served 500 users'."
      ]
    }
    """
    USER_MESSAGE = f"""
    ### CONTEXT
    **Relevance Score:** {relevence_score}/100
    
    **Resume Profile:**
    {json.dumps(resume_keys, indent=2)}
    
    **Job Description Profile:**
    {json.dumps(jd_keys, indent=2)}
    
    Explain the gaps that caused this score.
    """
    
    response=await client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role": "user", "content": USER_MESSAGE}
                  ],
        response_format={"type": "json_object"}
        
        )
    
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error parsing gap analysis: {e}")
        return {"error": "Failed to analyze gaps"}




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
        model=model,
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
    
    
    try:
        
        result=json.loads(response.choices[0].message.content)
        return jobData(**result)
    
    except json.JSONDecodeError:
        
        return jobData(
            role = "unknown",
            skills= [],
            is_valid_document = 0,
            summary = None
        )



async def LLM_distilliation_rich_user_data(text:str)->UserProfile:
    result=""
     
    output_example = {
        "is_valid_document":True,
        "name": "John Doe",
        "current_role": "Final Year Student",
        "years_of_experience": 0.5,
        "technical_skills": ["Python", "FastAPI", "React"],
        "project_complexity_level": "Intermediate",
        "domains_worked_in": ["Healthcare", "E-commerce"],
        "project_summaries": [
            "Built a Medical Image Tampering system using CNNs",
            "Developed a Portfolio site with React"
        ]
    }
     
    SYSTEM_PROMPT = f"""
    You are an expert Technical Recruiter. Your task is to extract user data ONLY from valid Resumes/CVs.
    
    ### STEP 1: VALIDATION (CRITICAL)
    Determine if the document is a Resume/CV.
    
    IT IS NOT A RESUME IF:
    - It is a research paper, abstract, or project report.
    - It is a news article, blog post, or tutorial.
    - It has sections like "Abstract", "Introduction", "Literature Review", "Conclusion".
    - It lists multiple authors (e.g., "By: Student A, Student B").
    
    IT IS A RESUME IF:
    - It describes ONE person's career history.
    - It has sections like "Experience", "Education", "Skills", "Projects".
    - It uses contact info (Email, Phone, LinkedIn) at the top.

    ### STEP 2: EXTRACTION
    IF VALID:
    1. Identify the user's name and current professional level.
    2. Extract ALL technical skills.
    3. Analyze projects for "Complexity Level" (Beginner/Intermediate/Advanced).
    4. Extract industries/domains.
    
    IF INVALID:
    Set "is_valid_document": false and leave other fields null/empty.

    Output strictly in valid JSON matching this structure:
    {json.dumps(output_example)}
    """
 
    client=AsyncOpenAI()
    
    response=await client.chat.completions.create(
        model=model,
        messages=[{
                "role": "system", 
                "content": f"{SYSTEM_PROMPT}"
            
            },
            {
                "role": "user", 
                "content": f"Extract key info:\n\n{text[:10000]}" 
            }],
        response_format={"type":"json_object"}
    )    
    
    try:
        
        result=json.loads(response.choices[0].message.content)
        return UserProfile(**result)
        
    except json.JSONDecodeError:
        
        return UserProfile(
            is_valid_document=False,
            name="Unknown", 
            current_role="Unknown", 
            years_of_experience=0, 
            technical_skills=[], 
            project_complexity_level="Beginner", 
            domains_worked_in=[],
            project_summaries=[]
        )        


async def career_roadmap_gen(resume_context:str, target_jobrole:str, hours_per_day:float):
    
    result=""
    
    output_example = {
        "target_role": "AI Engineer",
        "estimated_total_weeks": 12,
        "gap_analysis": {
            "missing_critical_skills": ["Vector DBs", "LangChain"],
            "skills_to_improve": ["Python Async"]
        },
        "roadmap": [
            {
                "phase_name": "Phase 1: Vector Search Foundations",
                "duration_weeks": 2,
                "goals": ["Understand Embeddings", "Build a Retriever"],
                "topics_to_cover": ["Cosine Similarity", "Pinecone/ChromaDB"],
                "project_idea": "Build a CLI tool to semantic search local PDFs",
                "project_complexity": "Intermediate",
                "resources": ["LangChain Docs", "Pinecone Academy"]
            }
        ]
    }
    
    SYSTEM_PROMPT = f"""
You are an elite Tech Career Coach. Your goal is to create a hyper-personalized learning roadmap to take a user from their current state to their Target Role.

### USER CONTEXT
- **Current Role:** {resume_context.current_role}
- **Experience:** {resume_context.years_of_experience} years
- **Current Skills:** {", ".join(resume_context.technical_skills)}
- **Completed Projects:** {"; ".join(resume_context.project_summaries)}
- **Project Complexity:** {resume_context.project_complexity_level}

### GOAL
- **Target Role:** {target_jobrole}
- **Time Commitment:** {hours_per_day} hours per day

### INSTRUCTIONS
1. **Gap Analysis:** Identify strictly what is missing between their Current Skills and the Target Role. Do NOT teach them what they already know.
2. **Project Strategy:** 
   - Look at their "Completed Projects". 
   - Do NOT suggest building the exact same things.
   - Suggest *evolutions* (e.g., if they built a basic chatbot, suggest adding RAG or Voice).
3. **Timeline:** Structure the roadmap into phases based on their `{hours_per_day} hours per day` constraint.
   - If they have few hours, stretch the timeline.
   - If they have many hours, condense it.
4. For resources, be SPECIFIC. Don't just say 'Youtube'. Say 'Andrej Karpathy's Neural Networks Zero to Hero'.



### OUTPUT FORMAT (JSON)
Output strictly in valid JSON matching this structure:
    {json.dumps(output_example)}
"""


    client=AsyncOpenAI()
    
    response=await client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":f"{SYSTEM_PROMPT}"},
            {"role": "user", "content": "Generate the roadmap now."}
        ],
        response_format={"type": "json_object"}
    )
    
    
    try:
        result=json.loads(response.choices[0].message.content)
        return CareerRoadmap(**result)
    
    except Exception as e:
        print(f"Error parsing roadmap: {e}")
        return e
    
    
async def fetch_leetcdoe_userdata(username)->LeetCodeStats:
    
    query = """
    query userProfile($username: String!) {
      matchedUser(username: $username) {
        submitStats: submitStatsGlobal {
          acSubmissionNum {
            difficulty
            count
          }
        }
        tagProblemCounts {
          advanced { tagName problemsSolved }
          intermediate { tagName problemsSolved }
          fundamental { tagName problemsSolved }
        }
      }
      recentAcSubmissionList(username: $username, limit: 15) {
        title
      }
    }
    """
    
    try:
        async with httpx.AsyncClient() as client:
            
            response=await client.post(
                "https://leetcode.com/graphql",
                json={"query": query, "variables": {"username": username}},
                headers={
                    "Content-Type": "application/json", 
                    "Referer": "https://leetcode.com",
                    "User-Agent": "Mozilla/5.0"
                },
                timeout=10.0
            )
            
            data=response.json()
            if "errors" in data or data.get("data", {}).get("matchedUser") is None:
                print("User not found or API error")
                return None
        
            user_data = data["data"]["matchedUser"]
            stats_list = user_data["submitStats"]["acSubmissionNum"]
            
            stats_map = {item["difficulty"]: item["count"] for item in stats_list}

            
            tags_map = {}
            for category in ["fundamental", "intermediate", "advanced"]:
                for tag_obj in user_data["tagProblemCounts"][category]:
                    tags_map[tag_obj["tagName"]] = tag_obj["problemsSolved"]
        
        # Recent List
            recent_list = [item["title"] for item in data["data"]["recentAcSubmissionList"]]
            
            return LeetCodeStats(
            total_solved=stats_map.get("All", 0),
            easy_solved=stats_map.get("Easy", 0),
            medium_solved=stats_map.get("Medium", 0),
            hard_solved=stats_map.get("Hard", 0),
            tag_counts=tags_map,
            recent_problems=recent_list
            )
        
            
    
    except Exception as e:
        print(f"LeetCode Fetch Error: {e}")
        return None 



async def suggested_questions(target_company,CATEGORY_MAP,COMPANY_GROUPS):
    
    SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_URL_HERE")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_KEY_HERE")
    
    supabase_client = await create_async_client(SUPABASE_URL, SUPABASE_KEY)
    target_input = target_company.lower().strip()
    
    selected_companies = []
    
    if "product" in target_input:
        selected_companies = COMPANY_GROUPS["product"]
    elif "service" in target_input:
        selected_companies = COMPANY_GROUPS["service"]
    elif "startup" in target_input:
        selected_companies = COMPANY_GROUPS["startup"]
    elif target_input in CATEGORY_MAP:
        selected_companies = [target_input]
    else:
        
        selected_companies = COMPANY_GROUPS["product"] 
    
    recommended_questions = []
    
    recommended_questions = []
    try:
        
        response = await supabase_client.rpc("get_top_questions", {
            "p_companies": selected_companies,
            "p_limit": 60
        }).execute()
        
        recommended_questions = response.data
    except Exception as e:
        print(f"Supabase RPC Error: {e}")
        
    return {
        "recommended_questions": recommended_questions
    }
    
    




async def DSA_roadmap_gen_llm(leetcode_data_of_user:str, target_company:str ,recommended_dict:dict,prep_months:int):
    
    total_weeks=prep_months*4
    
    recommended_list=recommended_dict["recommended_questions"]
    
    if prep_months <= 1:
        intensity = "High Intensity (Bootcamp)"
        problems_per_week = "15-20"
    elif prep_months <= 3:
        intensity = "Medium Intensity (Steady)"
        problems_per_week = "10-12"
    else:
        intensity = "Low Intensity (Marathon)"
        problems_per_week = "6-8"
    
    SYSTEM_PROMPT = f"""
    YYou are an elite Tech Interview Coach. Create a {total_weeks}-WEEK study plan ({intensity} Mode) for {target_company} based.
    
    ### STRICT CURRICULUM RULES:
    1. **Volume Requirement:** Assign **{problems_per_week} QUESTIONS per week**. (One per day + extras).
    2. **Pattern-Based Learning:** Focus on specific patterns (Sliding Window, DFS) per week.
       - Weeks 1-{total_weeks//3}: Focus on EASY/MEDIUM problems to build confidence.
       - Weeks {total_weeks//3}-{2*total_weeks//3}: Focus on MEDIUM/HARD company-specific problems.
       - Final Weeks: Mock Interviews & Hardest patterns (DP/Graph).
    3. **Gap Filling:** Prioritize patterns where the user's "Tag Counts" are low (from context).
    4. **Resource Specificity:** Recommend specific resources for that pattern (e.g., "NeetCode 150 Sliding Window video").

    ### OUTPUT FORMAT (JSON):
    {{
      "strategy_summary": "...",
      "weekly_plan": [
        {{
          "week": 1,
          "theme": "Pattern: Sliding Window & Two Pointers",
          "difficulty_focus": "Easy -> Medium",
          "goals": ["Master fixed vs variable window", "Understand shrinking window condition"],
          "questions": [
             {{ "title": "Maximum Subarray", "url": "...", "difficulty": "Easy", "pattern": "Kadane's Algo" }}
          ],
          "resources": "..."
        }}
      ]
    }}
    """
    user_stats_str = json.dumps(leetcode_data_of_user.model_dump(), indent=2) 
    
    user_message = f"""
    ### CONTEXT
    1. **Target Company:** {target_company}
    2. **User LeetCode Stats:** 
    {user_stats_str}
    3. **High-Value Recommended Questions (From DB):**
    {json.dumps(recommended_list[:50])} 
    
    Generate the roadmap JSON now.
    """

    client = AsyncOpenAI()
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {"error": "Failed to generate roadmap"}
    

