"""Transform API payloads into the model's expected feature matrix."""

import pandas as pd
from fastapi import HTTPException

from .config import (
    CODING_PLATFORMS,
    DEPARTMENT_FEATURES,
    REQUIRED_FIELDS,
    SCALED_FEATURES,
    VALID_DEPARTMENTS,
)
from .validation import to_binary, to_float, to_int


def get_coding_skill_score(platform: str, rating: float) -> int:
    thresholds = {
        "Codeforces": (1200, 1600, 2100),
        "CodeChef": (1400, 1800, 2200),
        "LeetCode": (1500, 1900, 2301),
        "HackerRank": (1600, 2001, 2401),
    }
    platform_thresholds = thresholds.get(platform)
    if platform_thresholds is None:
        return 0
    first, second, third = platform_thresholds
    if rating < first:
        return 25
    if rating < second:
        return 50
    if rating < third:
        return 75
    return 100


def build_model_input(raw: dict, model_features: list[str], scaler) -> pd.DataFrame:
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
            detail=(
                f"Invalid coding_contest_platform '{platform}'. "
                f"Expected one of: {sorted(CODING_PLATFORMS)}"
            ),
        )

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
        "coding_contest_skill_score": get_coding_skill_score(
            platform, to_float(raw["coding_contest_rating"], "coding_contest_rating")
        ),
    }

    model_input.update({feature: 0 for feature in DEPARTMENT_FEATURES})
    if department != "B.B.A":
        model_input[f"department_{department}"] = 1

    dataframe = pd.DataFrame([model_input], columns=model_features)
    dataframe[list(SCALED_FEATURES)] = scaler.transform(dataframe[list(SCALED_FEATURES)])
    return dataframe
