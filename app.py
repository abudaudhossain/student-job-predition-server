from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")
MODEL_FEATURES = list(model.feature_names_in_)

DEPARTMENT_FEATURES = [
    "Department_CSE",
    "Department_EEE",
    "Department_IT",
    "Department_Management",
    "Department_SWE",
]

LANGUAGE_FEATURES = ["C", "C++", "Java", "JavaScript", "PHP", "Python"]

YES_NO_MAP = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}

INPUT_KEYS = {
    "department": ["department", "Department"],
    "age": ["age", "Age"],
    "cgpa": ["cgpa", "CGPA"],
    "problems_solved": [
        "problems_solved",
        "Approximate Number of Problems Solved on Online Judge Platforms",
    ],
    "technical_skill": ["technical_skill", "Self-Rated Technical Skill Level"],
    "communication_skill": ["communication_skill", "Communication Skill Level"],
    "problem_solving_ability": [
        "problem_solving_ability",
        "problem_solving",
        "Real-World Problem Solving Ability",
    ],
    "teamwork": ["teamwork", "Teamwork Ability"],
    "extra_activities": ["extra_activities", "Number of Extra-Curricular Activities"],
    "projects_completed": ["projects_completed", "Number of Completed Projects"],
    "internship_experience": ["internship_experience", "Internship Experience"],
    "gender": ["gender", "Gender"],
    "completed_extra_courses": [
        "completed_extra_courses",
        "How many skill development courses have you completed outside your academic curriculum?",
    ],
    "published_projects": [
        "published_projects",
        "Have you published projects on GitHub or deployed online?",
    ],
    "job_by_referral": [
        "job_by_referral",
        "Was the job obtained through networking or referral?",
    ],
    "education_aligned_job": [
        "education_aligned_job",
        "Does your current employment align with your educational background?",
    ],
    "programming_languages": ["programming_languages", "Programming Languages You Know"],
    "activities": [
        "activities",
        "Which of the following extra-curricular or technical activities have you participated in during your university studies?",
    ],
}

# Approximated from notebook-style activity scoring.
ACTIVITY_SCORE_MAP = {
    "Programming Contests": 5,
    "Hackathons": 5,
    "Open Source Contribution": 5,
    "Freelancing": 4,
    "Workshops": 3,
    "Seminars": 2,
    "None": 0,
    "Unknown": 0,
}


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


def pick_value(raw: dict, keys: list[str], field_name: str, required: bool = False):
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    if required:
        raise HTTPException(status_code=400, detail=f"Missing required field: {field_name}")
    return None


def parse_languages(languages_raw) -> tuple[dict, int]:
    if isinstance(languages_raw, list):
        items = [str(item).strip() for item in languages_raw]
    else:
        items = [part.strip() for part in str(languages_raw).split(",")]
    selected = {item for item in items if item}

    lang_values = {lang: 1 if lang in selected else 0 for lang in LANGUAGE_FEATURES}
    return lang_values, sum(lang_values.values())


def compute_activity_score(activity_raw, fallback_count: int) -> int:
    if activity_raw is None:
        return fallback_count

    if isinstance(activity_raw, list):
        activities = [str(item).strip() for item in activity_raw if str(item).strip()]
    else:
        activities = [part.strip() for part in str(activity_raw).split(",") if part.strip()]

    if not activities:
        return fallback_count

    scores = [ACTIVITY_SCORE_MAP.get(activity, 0) for activity in activities]
    max_score = max(scores) if scores else 0
    return max_score if max_score > 0 else fallback_count


def build_model_input(raw: dict) -> dict:
    dept = str(pick_value(raw, INPUT_KEYS["department"], "department", required=True)).strip()

    model_input = {
        "Age": to_int(pick_value(raw, INPUT_KEYS["age"], "age"), "age"),
        "CGPA": to_float(pick_value(raw, INPUT_KEYS["cgpa"], "cgpa"), "cgpa"),
        "Approximate Number of Problems Solved on Online Judge Platforms": to_int(
            pick_value(raw, INPUT_KEYS["problems_solved"], "problems_solved"),
            "problems_solved",
        ),
        "Self-Rated Technical Skill Level": to_int(
            pick_value(raw, INPUT_KEYS["technical_skill"], "technical_skill"), "technical_skill"
        ),
        "Communication Skill Level": to_int(
            pick_value(raw, INPUT_KEYS["communication_skill"], "communication_skill"), "communication_skill"
        ),
        "Real-World Problem Solving Ability": to_int(
            pick_value(raw, INPUT_KEYS["problem_solving_ability"], "problem_solving_ability"),
            "problem_solving_ability",
        ),
        "Teamwork Ability": to_int(pick_value(raw, INPUT_KEYS["teamwork"], "teamwork"), "teamwork"),
        "Number of Extra-Curricular Activities": to_int(
            pick_value(raw, INPUT_KEYS["extra_activities"], "extra_activities"), "extra_activities"
        ),
        "Number of Completed Projects": to_int(
            pick_value(raw, INPUT_KEYS["projects_completed"], "projects_completed"), "projects_completed"
        ),
        "Internship Experience": to_binary(
            pick_value(raw, INPUT_KEYS["internship_experience"], "internship_experience"),
            "internship_experience",
        ),
        "Gender_Male": 1 if str(pick_value(raw, INPUT_KEYS["gender"], "gender") or "").strip().lower() == "male" else 0,
        "completed_extra_coures": to_int(
            pick_value(raw, INPUT_KEYS["completed_extra_courses"], "completed_extra_courses"),
            "completed_extra_courses",
        ),
        "published projects": to_binary(
            pick_value(raw, INPUT_KEYS["published_projects"], "published_projects"),
            "published_projects",
        ),
        "job obtained by referral": to_binary(
            pick_value(raw, INPUT_KEYS["job_by_referral"], "job_by_referral"),
            "job_by_referral",
        ),
        "educational background job": to_binary(
            pick_value(raw, INPUT_KEYS["education_aligned_job"], "education_aligned_job"),
            "education_aligned_job",
        ),
    }

    for dept_feature in DEPARTMENT_FEATURES:
        model_input[dept_feature] = 0
    mapped_dept_key = f"Department_{dept}"
    if mapped_dept_key in model_input:
        model_input[mapped_dept_key] = 1

    language_values, language_count = parse_languages(
        pick_value(raw, INPUT_KEYS["programming_languages"], "programming_languages") or ""
    )
    model_input.update(language_values)
    model_input["programing_skill_count"] = language_count

    model_input["activity_score"] = compute_activity_score(
        pick_value(raw, INPUT_KEYS["activities"], "activities"),
        model_input["Number of Extra-Curricular Activities"],
    )

    return {feature: model_input.get(feature, 0) for feature in MODEL_FEATURES}

@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: dict):
    model_input = build_model_input(data)
    df = pd.DataFrame([model_input], columns=MODEL_FEATURES)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "status": "Got a Job" if pred == 1 else "Did Not Get a Job",
        "confidence": float(prob)
    }