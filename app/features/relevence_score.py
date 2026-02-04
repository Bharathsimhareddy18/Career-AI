from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def relevence_score_function(resume_vectors:list,JD_vectors:list):
    
    res_array = np.array(resume_vectors).reshape(1, -1)
    jd_array = np.array(JD_vectors).reshape(1, -1)
    
    score = cosine_similarity(res_array, jd_array)[0][0]
    
    return score
    
