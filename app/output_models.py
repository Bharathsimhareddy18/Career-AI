from pydantic import BaseModel,Field
from typing import List, Optional,Dict

class jobData(BaseModel):
    role: Optional[str] 
    skills: Optional[List[str]]
    is_valid_document: bool 
    summary: Optional[str] = None
    

class UserProfile(BaseModel):
    is_valid_document: bool = Field(default=True, description="True if a valid resume was parsed")
    name : Optional[str] =Field(default=None, description="User's full name")
    current_role : Optional[str] = Field(default=None ,description="Inferred current role (e.g., Student, Frontend Dev)")
    years_of_experience: Optional[float] = Field(default=0.0,description="Total years of professional experience")
    technical_skills: Optional[List[str]] = Field(default_factory=list, description="List of hard technical skills")
    project_complexity_level: Optional[str] = Field(default=None, description="Project complexity")
    domains_worked_in: Optional[List[str]] = Field(default_factory=list, description="List of industries")
    project_summaries: Optional[List[str]] = Field(default_factory=list, description="Brief summaries of key projects")
    
    

class GapAnalysis(BaseModel):
    missing_critical_skills: List[str] = Field(description="Skills required for target role that user lacks completely")
    skills_to_improve: List[str] = Field(description="Skills user has but needs to deepen for the target role")
    
class RoadmapPhase(BaseModel):
    phase_name: str = Field(description="e.g., 'Phase 1: Foundation Gaps'")
    duration_weeks: int = Field(description="Estimated number of weeks for this phase")
    goals: List[str] = Field(description="High-level objectives for this phase")
    topics_to_cover: List[str] = Field(description="Specific technical topics (e.g., 'Asyncio', 'Docker Networking')")
    project_idea: str = Field(description="A specific, hands-on project description that applies these skills")
    project_complexity: str = Field(description="Complexity of the suggested project (e.g., 'Intermediate')")
    resources: List[str] = Field(description="Specific, high-quality resources. Prefer exact book titles, specific Coursera course names, or direct documentation links (e.g., 'HuggingFace Tokenizers Docs' instead of just 'HuggingFace').")   
     
class CareerRoadmap(BaseModel):
    target_role: str = Field(description="The role the user is aiming for")
    estimated_total_weeks: int = Field(description="Total duration of the roadmap")
    gap_analysis: GapAnalysis
    roadmap: List[RoadmapPhase]
    
    
class LeetCodeStats(BaseModel):
    total_solved: int
    easy_solved: int
    medium_solved: int
    hard_solved: int
    tag_counts: Dict[str, int]
    recent_problems: List[str]
    
    
    
class leetcode_user(BaseModel):
    leetcode_public:str=Field(description="user leetcode profile link")
    user_target_company:str=Field(description="user target company")
    time_period_for_interview:int=Field(description="no of months for prep")

class LeetcodeRoadmap(BaseModel):
    user_target_company:str=Field(description="User choice of company or type of company")
        
    