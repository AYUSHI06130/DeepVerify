def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, round(v)))

def score_from_evidence(evidence):
    if not evidence:
        return 50
    total = sum(x.get("weight", 1) for x in evidence)
    risk = sum(x.get("risk", 0) * x.get("weight", 1) for x in evidence) / total
    return clamp(100 - risk)

def label(score):
    if score >= 75: return "Likely trustworthy"
    if score >= 50: return "Needs verification"
    if score >= 25: return "Suspicious"
    return "High-risk / likely manipulated"