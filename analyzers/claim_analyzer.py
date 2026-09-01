import os
import requests
from urllib.parse import urlparse
from .scoring import score_from_evidence, label

URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
KNOWN_HIGH_TRUST = {"reuters.com","apnews.com","bbc.com","theguardian.com",
                    "nytimes.com","who.int","un.org","gov.in"}

def domain_from_text(text):
    for token in text.split():
        if token.startswith(("http://","https://")):
            return urlparse(token).netloc.lower().replace("www.","")
    return None

def factcheck_search(claim):
    key = os.getenv("FACTCHECK_API_KEY")
    if not key:
        return []
    r = requests.get(URL, params={"query":claim,"pageSize":5,"key":key}, timeout=10)
    r.raise_for_status()
    return r.json().get("claims", [])

def analyze_claim(claim):
    evidence = []
    ratings = []
    try:
        claims = factcheck_search(claim)
        if claims:
            for c in claims[:5]:
                for review in c.get("claimReview", []):
                    ratings.append({
                        "publisher":review.get("publisher",{}).get("name"),
                        "rating":review.get("textualRating"),
                        "title":review.get("title"),
                        "url":review.get("url")
                    })
            evidence.append({
                "name":"Existing fact checks found","risk":55,"weight":0.85,
                "detail":f"Found {len(ratings)} fact-check review(s). Read the reviews before deciding."
            })
        else:
            evidence.append({
                "name":"No matching fact check found","risk":5,"weight":0.2,
                "detail":"No matching review was returned. That does not mean the claim is true."
            })
    except Exception as e:
        evidence.append({
            "name":"Fact-check lookup unavailable","risk":10,"weight":0.15,
            "detail":str(e)
        })

    domain = domain_from_text(claim)
    if domain in KNOWN_HIGH_TRUST:
        evidence.append({
            "name":"Recognized source domain","risk":5,"weight":0.35,
            "detail":f"{domain} is in this prototype's small baseline source list."
        })
    elif domain:
        evidence.append({
            "name":"Source not in baseline list","risk":20,"weight":0.25,
            "detail":f"{domain} is not in the small baseline list; that is not proof it is unreliable."
        })
    else:
        evidence.append({
            "name":"No source URL detected","risk":15,"weight":0.25,
            "detail":"No URL was detected, so source context could not be assessed."
        })

    trust = score_from_evidence(evidence)
    return {
        "type":"claim","trust_score":trust,"label":label(trust),
        "summary":"Claim verification combines existing fact-check matches and basic source context; it does not independently prove truth.",
        "claim":claim,"factchecks":ratings,"evidence":evidence
    }