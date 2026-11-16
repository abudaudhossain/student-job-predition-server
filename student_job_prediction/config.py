"""Application configuration and model feature definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

CODING_PLATFORMS = frozenset({"Codeforces", "CodeChef", "LeetCode", "HackerRank"})
VALID_DEPARTMENTS = frozenset({"B.B.A", "CSE", "Civil", "EEE", "English", "Pharmacy"})

DEPARTMENT_FEATURES = (
    "department_CSE",
    "department_Civil",
    "department_EEE",
    "department_English",
    "department_Pharmacy",
)

SCALED_FEATURES = (
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
)

REQUIRED_FIELDS = (
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
)

YES_NO_MAP = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
