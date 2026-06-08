from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
MODEL_FEATURES = list(model.feature_names_)

CODING_PLATFORMS = {"Codeforces", "CodeChef", "LeetCode", "HackerRank"}
VALID_DEPARTMENTS = {"B.B.A", "CSE", "Civil", "EEE", "English", "Pharmacy"}
DEPARTMENT_FEATURES = [
    "department_CSE",
    "department_Civil",
    "department_EEE",
    "department_English",
    "department_Pharmacy",
]

SCALED_FEATURES = [
    "cgpa",
    "programming_skill_score",
    "problem_solving_score",
    "database_skill_score",
    "internships_count",
    "hackathons_participated",
    "certifications_count",
    "projects_count",
    "github_repos",
    "communication_skill_score",
    "teamwork_score",
    "learning_consistency_score",
    "aptitude_test_score",
    "mock_interview_score",
    "resume_quality_score",
    "leadership_score",
    "extracurricular_score",
    "presentation_skill_score",
    "coding_contest_skill_score",
]

REQUIRED_FIELDS = [
    "cgpa",
    "department",
    "programming_skill_score",
    "problem_solving_score",
    "database_skill_score",
    "coding_contest_rating",
    "coding_contest_platform",
    "internships_count",
    "hackathons_participated",
    "freelance_experience",
    "certifications_count",
    "projects_count",
    "github_repos",
    "communication_skill_score",
    "teamwork_score",
    "learning_consistency_score",
    "aptitude_test_score",
    "mock_interview_score",
    "resume_quality_score",
    "leadership_score",
    "extracurricular_score",
    "presentation_skill_score",
]

YES_NO_MAP = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}


def to_binary(value, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in YES_NO_MAP:
            return YES_NO_MAP[normalized]
    raise HTTPException(status_code=400, detail=f"Invalid value for '{field_name}': {value}")


def to_int(value, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid integer for '{field_name}': {value}")


def to_float(value, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid number for '{field_name}': {value}")


def get_coding_skill_score(platform: str, rating: float) -> int:
    score = 25
    if platform == "Codeforces":
        if rating < 1200:
            return score
        if rating <= 1599:
            return 2 * score
        if rating <= 2099:
            return 3 * score
        return 4 * score
    if platform == "CodeChef":
        if rating < 1400:
            return score
        if rating <= 1799:
            return 2 * score
        if rating <= 2199:
            return 3 * score
        return 4 * score
    if platform == "LeetCode":
        if rating < 1500:
            return score
        if rating <= 1899:
            return 2 * score
        if rating <= 2300:
            return 3 * score
        return 4 * score
    if platform == "HackerRank":
        if rating < 1600:
            return score
        if rating <= 2000:
            return 2 * score
        if rating <= 2400:
            return 3 * score
        return 4 * score
    return 0


def build_model_input(raw: dict) -> pd.DataFrame:
    for field in REQUIRED_FIELDS:
        if field not in raw or raw[field] is None:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    department = str(raw["department"]).strip()
    if department not in VALID_DEPARTMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department '{department}'. Expected one of: {sorted(VALID_DEPARTMENTS)}",
        )

    platform = str(raw["coding_contest_platform"]).strip()
    if platform not in CODING_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid coding_contest_platform '{platform}'. Expected one of: {sorted(CODING_PLATFORMS)}",
        )

    rating = to_float(raw["coding_contest_rating"], "coding_contest_rating")
    model_input = {
        "cgpa": to_float(raw["cgpa"], "cgpa"),
        "programming_skill_score": to_int(raw["programming_skill_score"], "programming_skill_score"),
        "problem_solving_score": to_int(raw["problem_solving_score"], "problem_solving_score"),
        "database_skill_score": to_int(raw["database_skill_score"], "database_skill_score"),
        "internships_count": to_int(raw["internships_count"], "internships_count"),
        "hackathons_participated": to_int(raw["hackathons_participated"], "hackathons_participated"),
        "freelance_experience": to_binary(raw["freelance_experience"], "freelance_experience"),
        "certifications_count": to_int(raw["certifications_count"], "certifications_count"),
        "projects_count": to_int(raw["projects_count"], "projects_count"),
        "github_repos": to_int(raw["github_repos"], "github_repos"),
        "communication_skill_score": to_int(raw["communication_skill_score"], "communication_skill_score"),
        "teamwork_score": to_int(raw["teamwork_score"], "teamwork_score"),
        "learning_consistency_score": to_int(raw["learning_consistency_score"], "learning_consistency_score"),
        "aptitude_test_score": to_int(raw["aptitude_test_score"], "aptitude_test_score"),
        "mock_interview_score": to_int(raw["mock_interview_score"], "mock_interview_score"),
        "resume_quality_score": to_int(raw["resume_quality_score"], "resume_quality_score"),
        "leadership_score": to_int(raw["leadership_score"], "leadership_score"),
        "extracurricular_score": to_int(raw["extracurricular_score"], "extracurricular_score"),
        "presentation_skill_score": to_int(raw["presentation_skill_score"], "presentation_skill_score"),
        "coding_contest_skill_score": get_coding_skill_score(platform, rating),
    }

    for dept_feature in DEPARTMENT_FEATURES:
        model_input[dept_feature] = 0
    if department != "B.B.A":
        model_input[f"department_{department}"] = 1

    df = pd.DataFrame([model_input], columns=MODEL_FEATURES)
    df[SCALED_FEATURES] = scaler.transform(df[SCALED_FEATURES])
    return df


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: dict):
    df = build_model_input(data)
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])

    return {
        "status": "Got a Job" if pred == 1 else "Did Not Get a Job",
        "confidence": prob,
    }
