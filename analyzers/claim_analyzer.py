import os
import re
import requests
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

from .scoring import score_from_evidence, label


URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


KNOWN_HIGH_TRUST = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "theguardian.com",
    "nytimes.com",
    "who.int",
    "un.org",
    "gov.in",
    "nasa.gov",
}


# ---------------------------------------------------------
# SOURCE / URL HELPERS
# ---------------------------------------------------------

def domain_from_text(text):
    """
    Extract the first URL domain from the claim.
    """

    match = re.search(
        r"https?://[^\s]+",
        text
    )

    if not match:
        return None

    try:
        domain = urlparse(match.group(0)).netloc.lower()
        return domain.replace("www.", "")
    except Exception:
        return None


# ---------------------------------------------------------
# FACT CHECK API
# ---------------------------------------------------------

def factcheck_search(claim):
    """
    Search Google's Fact Check Tools API.
    """

    key = os.getenv("FACTCHECK_API_KEY")

    if not key:
        return []

    response = requests.get(
        URL,
        params={
            "query": claim,
            "pageSize": 10,
            "key": key
        },
        timeout=10
    )

    response.raise_for_status()

    return response.json().get("claims", [])


# ---------------------------------------------------------
# RATING INTERPRETATION
# ---------------------------------------------------------

def rating_to_risk(rating):
    """
    Convert common fact-check ratings into risk.

    0   = low misinformation risk
    100 = high misinformation risk
    """

    if not rating:
        return 50

    rating = rating.lower().strip()

    # Check more specific phrases FIRST

    if any(word in rating for word in [
        "pants on fire",
        "fabricated",
        "completely false",
        "false",
        "fake",
        "incorrect",
        "not true"
    ]):
        return 95

    if any(word in rating for word in [
        "mostly false",
        "largely false",
        "false claim"
    ]):
        return 80

    if any(word in rating for word in [
        "misleading",
        "misrepresents",
        "misrepresented",
        "out of context",
        "missing context",
        "distorts the facts"
    ]):
        return 75

    if any(word in rating for word in [
        "half true",
        "partly true",
        "mixed"
    ]):
        return 50

    if any(word in rating for word in [
        "mostly true",
        "largely true"
    ]):
        return 25

    if any(word in rating for word in [
        "true",
        "correct",
        "accurate"
    ]):
        return 10

    return 50


# ---------------------------------------------------------
# DETECT WHETHER FACT CHECK SUPPORTS OR CONTRADICTS CLAIM
# ---------------------------------------------------------

def normalize_text(text):
    """
    Normalize text for simple comparison.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def claim_similarity(user_claim, checked_claim):
    """
    Very simple word-overlap similarity.

    This is not an AI semantic model.
    It is only used to determine whether
    a returned fact-check is reasonably related.
    """

    a = set(normalize_text(user_claim).split())
    b = set(normalize_text(checked_claim).split())

    if not a or not b:
        return 0

    return len(a & b) / len(a)

def get_claim_polarity(text):
    """
    Estimate whether a claim is positive, negative, or neutral.

    This is a lightweight rule-based check used to distinguish
    a claim from a fact-checked claim making the opposite assertion.
    """

    text = normalize_text(text)

    negative_patterns = [
        "not",
        "no",
        "never",
        "false",
        "does not",
        "do not",
        "did not",
        "has not",
        "have not",
        "cannot",
        "cant",
        "doesnt",
        "dont",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "unlikely",
        "impossible",
        "myth",
        "debunked"
    ]

    positive_patterns = [
        "is",
        "are",
        "was",
        "were",
        "does",
        "do",
        "did",
        "causes",
        "cause",
        "increases",
        "increased",
        "found",
        "discovered",
        "supports",
        "confirmed"
    ]

    negative_score = 0
    positive_score = 0

    words = text.split()

    for word in words:
        if word in negative_patterns:
            negative_score += 1

    for word in words:
        if word in positive_patterns:
            positive_score += 1

    if negative_score > positive_score:
        return -1

    if positive_score > negative_score:
        return 1

    return 0

def is_opposite_claim(user_claim, checked_claim):
    """
    Detect whether the fact-checked claim makes the opposite
    assertion to the user's claim.
    """

    user = normalize_text(user_claim)
    checked = normalize_text(checked_claim)

    # -----------------------------------------
    # Common explicit opposite constructions
    # -----------------------------------------

    opposite_pairs = [

        # Cause / no cause
        (" causes ", " does not cause "),
        (" cause ", " do not cause "),
        (" causes ", " does not "),
        (" cause ", " do not "),

        # IS / IS NOT
        (" is ", " is not "),
        (" are ", " are not "),
        (" was ", " was not "),
        (" were ", " were not "),

        # CAN / CANNOT
        (" can ", " cannot "),
        (" can ", " can't "),

        # WILL / WILL NOT
        (" will ", " will not "),
        (" will ", " won't "),

        # FOUND / DID NOT FIND
        (" found ", " did not find "),
        (" found ", " has not found "),
        (" found ", " have not found "),

        # DISCOVERED / DID NOT DISCOVER
        (" discovered ", " did not discover "),
        (" discovered ", " has not discovered "),
        (" discovered ", " have not discovered "),
    ]

    user_padded = f" {user} "
    checked_padded = f" {checked} "

    for positive, negative in opposite_pairs:

        if (
            positive in user_padded
            and negative in checked_padded
        ):
            return True

        if (
            negative in user_padded
            and positive in checked_padded
        ):
            return True

    # -----------------------------------------
    # Handle phrases such as:
    #
    # "No one has shown that..."
    #
    # This is a negative assertion even though
    # the word "not" isn't present.
    # -----------------------------------------

    negative_phrases = [
        "no one",
        "nobody",
        "nothing",
        "has not",
        "have not",
        "does not",
        "do not",
        "did not",
        "is not",
        "are not",
        "was not",
        "were not",
        "cannot",
        "can't",
        "never",
        "not true",
        "not caused",
        "not cause",
        "not responsible",
        "is a myth",
        "are a myth"
    ]

    user_negative = any(
        phrase in user_padded
        for phrase in negative_phrases
    )

    checked_negative = any(
        phrase in checked_padded
        for phrase in negative_phrases
    )

    similarity = claim_similarity(
        user_claim,
        checked_claim
    )

    # If one claim is clearly negative and the other isn't,
    # and the claims share enough vocabulary, treat them as
    # opposite assertions.
    if (
        similarity >= 0.25
        and user_negative != checked_negative
    ):
        return True

    return False

# ---------------------------------------------------------
# MAIN CLAIM ANALYZER
# ---------------------------------------------------------

def analyze_claim(claim):

    claim = claim.strip()

    if not claim:
        raise ValueError("Enter a claim first.")

    evidence = []
    ratings = []

    # -----------------------------------------------------
    # 1. FACT CHECK SEARCH
    # -----------------------------------------------------

    api_available = bool(
        os.getenv("FACTCHECK_API_KEY")
    )

    if not api_available:

        evidence.append({
            "name": "Fact-check service unavailable",
            "risk": 50,
            "weight": 0.8,
            "detail": (
                "No FACTCHECK_API_KEY is configured. "
                "DeepVerify cannot verify this claim against "
                "external fact-check databases."
            )
        })

    else:

        try:

            claims = factcheck_search(claim)

            if claims:

                review_risks = []

                for c in claims[:10]:

                    checked_claim = c.get(
                        "text",
                        ""
                    )

                    similarity = claim_similarity(
                        claim,
                        checked_claim
                    )

                    opposite = is_opposite_claim(
                        claim,
                        checked_claim
                    )

                    for review in c.get(
                        "claimReview",
                        []
                    ):

                        publisher = (
                            review
                            .get("publisher", {})
                            .get("name")
                        )

                        rating = review.get(
                            "textualRating"
                        )

                        title = review.get(
                            "title"
                        )

                        url = review.get(
                            "url"
                        )

                        raw_risk = rating_to_risk(
                            rating
                        )

                        # -------------------------------------------------
                        # IMPORTANT:
                        # If Google fact-checked the OPPOSITE claim,
                        # invert the rating.
                        #
                        # Example:
                        #
                        # Checked claim:
                        # "Humans do NOT cause global warming"
                        #
                        # Rating:
                        # "False"
                        #
                        # Therefore user's positive claim is supported.
                        # -------------------------------------------------

                        if opposite:

                            if raw_risk >= 80:
                                risk = 10
                            elif raw_risk >= 65:
                                risk = 25
                            elif raw_risk >= 45:
                                risk = 50
                            else:
                                risk = 85

                            relationship = "opposite"

                        else:

                            risk = raw_risk
                            relationship = "related"

                        # Only use reasonably related results
                        if similarity >= 0.25:

                            review_risks.append(
                                risk
                            )

                            ratings.append({
                                "publisher": publisher,
                                "rating": rating,
                                "title": title,
                                "url": url,
                                "checked_claim": checked_claim,
                                "relationship": relationship
                            })

                # -------------------------------------------------
                # FACT-CHECK EVIDENCE
                # -------------------------------------------------

                if review_risks:

                    avg_risk = (
                        sum(review_risks)
                        / len(review_risks)
                    )

                    review_count = len(
                        review_risks
                    )

                    if review_count >= 3:
                        weight = 1.25
                    elif review_count == 2:
                        weight = 1.10
                    else:
                        weight = 0.95

                    evidence.append({
                        "name": "Fact-check evidence",
                        "risk": round(
                            avg_risk,
                            2
                        ),
                        "weight": weight,
                        "detail": (
                            f"Found {review_count} related fact-check review(s). "
                            f"DeepVerify compared the submitted claim with "
                            f"the fact-checked claims and accounted for "
                            f"opposing claim wording where detected."
                        )
                    })

                else:

                    evidence.append({
                        "name": "No sufficiently related fact check",
                        "risk": 50,
                        "weight": 0.8,
                        "detail": (
                            "The fact-check service returned "
                            "results, but none were sufficiently "
                            "related to the submitted claim."
                        )
                    })

            else:

                evidence.append({
                    "name": "No matching fact check found",
                    "risk": 50,
                    "weight": 0.8,
                    "detail": (
                        "No matching fact-check review was found. "
                        "This does NOT mean the claim is true; "
                        "the claim remains unverified."
                    )
                })

        except Exception as e:

            evidence.append({
                "name": "Fact-check lookup failed",
                "risk": 50,
                "weight": 0.8,
                "detail": (
                    "The external fact-check service could "
                    "not be reached. The claim was therefore "
                    "not treated as trustworthy."
                )
            })

    # -----------------------------------------------------
    # 2. SOURCE CONTEXT
    # -----------------------------------------------------

    domain = domain_from_text(claim)

    if domain in KNOWN_HIGH_TRUST:

        evidence.append({
            "name": "Recognized source domain",
            "risk": 20,
            "weight": 0.25,
            "detail": (
                f"{domain} is a recognized source domain "
                f"in DeepVerify's baseline list. "
                f"This supports source credibility but "
                f"does not prove the claim itself."
            )
        })

    elif domain:

        evidence.append({
            "name": "Source domain detected",
            "risk": 45,
            "weight": 0.20,
            "detail": (
                f"The claim references {domain}. "
                f"The domain is not in DeepVerify's "
                f"small baseline list, so additional "
                f"verification is recommended."
            )
        })

    else:

        evidence.append({
            "name": "No source URL detected",
            "risk": 50,
            "weight": 0.15,
            "detail": (
                "No source URL was detected. "
                "Source credibility could not "
                "be assessed."
            )
        })

    # -----------------------------------------------------
    # 3. CALCULATE TRUST SCORE
    # -----------------------------------------------------

    trust = score_from_evidence(
        evidence
    )

    # -----------------------------------------------------
    # 4. EXPLICIT CLAIM VERDICT
    # -----------------------------------------------------

    if ratings:

        risks = [
            rating_to_risk(
                r.get("rating")
            )
            for r in ratings
            if r.get("rating")
        ]

        # IMPORTANT:
        # Recalculate using the relationship-aware
        # risk stored while processing the reviews.

        relationship_risks = []

        for r in ratings:

            raw_risk = rating_to_risk(
                r.get("rating")
            )

            if r.get("relationship") == "opposite":

                if raw_risk >= 80:
                    raw_risk = 10
                elif raw_risk >= 65:
                    raw_risk = 25
                elif raw_risk >= 45:
                    raw_risk = 50
                else:
                    raw_risk = 85

            relationship_risks.append(
                raw_risk
            )

        if relationship_risks:

            avg_rating_risk = (
                sum(relationship_risks)
                / len(relationship_risks)
            )

            if avg_rating_risk >= 80:
                verdict = "Likely false"

            elif avg_rating_risk >= 65:
                verdict = "Likely misleading"

            elif avg_rating_risk >= 45:
                verdict = "Mixed / disputed"

            elif avg_rating_risk <= 20:
                verdict = (
                    "Supported by available fact checks"
                )

            else:
                verdict = "Mostly supported"

        else:

            verdict = "Unverified"

    else:

        # NO FACT CHECK = NOT TRUE
        verdict = "Unverified"

    # -----------------------------------------------------
    # 5. RETURN RESULT
    # -----------------------------------------------------

    return {
        "type": "claim",
        "trust_score": trust,
        "label": label(trust),
        "verdict": verdict,
        "summary": (
            "DeepVerify compares the claim against "
            "available fact-check evidence and source "
            "context. A lack of fact-check results "
            "does not mean the claim is true."
        ),
        "claim": claim,
        "factchecks": ratings,
        "evidence": evidence
    }