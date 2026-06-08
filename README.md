# Student Job Prediction Server

This project is a FastAPI-based ML inference server that predicts whether a student is likely to be placed in a job based on academic, technical, and employability-related inputs.

The server loads a CatBoost classifier from `model.pkl` and applies the same preprocessing pipeline used in `job_prediction_model.py` before running inference.

## About the server

- Framework: `FastAPI`
- Model: `CatBoostClassifier` (loaded with `joblib`)
- Preprocessing: `pandas`, `scikit-learn` (`StandardScaler`)
- Main endpoints:
  - `GET /` -> health check message
  - `POST /predict` -> placement prediction and confidence score

`POST /predict` returns:
- `status`: `"Got a Job"` or `"Did Not Get a Job"`
- `confidence`: probability score for the positive class (placed)

## Preprocessing flow

Before prediction, the API:

1. Derives `coding_contest_skill_score` from `coding_contest_rating` and `coding_contest_platform`
2. One-hot encodes `department` (`B.B.A` is the reference category)
3. Scales numeric features using `scaler.pkl`
4. Runs inference with the trained CatBoost model

## Requirements

- Python 3.9+ recommended
- Files required in project root:
  - `app.py`
  - `model.pkl`
  - `scaler.pkl`
  - `requirements.txt`

## How to run

### 1) Create and activate a virtual environment

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Server will run at:

- `http://127.0.0.1:8000`

## API usage

### Health check

Request:

```http
GET /
```

Response:

```json
{
  "message": "API is running"
}
```

### Predict job outcome

Request:

```http
POST /predict
Content-Type: application/json
```

Example body:

```json
{
  "cgpa": 4.0,
  "department": "CSE",
  "programming_skill_score": 100,
  "problem_solving_score": 95,
  "database_skill_score": 86,
  "coding_contest_rating": 1898,
  "coding_contest_platform": "HackerRank",
  "internships_count": 3,
  "hackathons_participated": 5,
  "freelance_experience": 0,
  "certifications_count": 4,
  "projects_count": 5,
  "github_repos": 20,
  "communication_skill_score": 92,
  "teamwork_score": 75,
  "learning_consistency_score": 97,
  "aptitude_test_score": 100,
  "mock_interview_score": 97,
  "resume_quality_score": 100,
  "leadership_score": 95,
  "extracurricular_score": 96,
  "presentation_skill_score": 87
}
```

Example response:

```json
{
  "status": "Got a Job",
  "confidence": 0.9919
}
```

## Request fields

All fields below are required.

| Field | Type | Notes |
| --- | --- | --- |
| `cgpa` | number | Student CGPA |
| `department` | string | One of `B.B.A`, `CSE`, `Civil`, `EEE`, `English`, `Pharmacy` |
| `programming_skill_score` | integer | Self-rated programming skill score |
| `problem_solving_score` | integer | Problem solving score |
| `database_skill_score` | integer | Database skill score |
| `coding_contest_rating` | number | Rating on the selected coding platform |
| `coding_contest_platform` | string | One of `Codeforces`, `CodeChef`, `LeetCode`, `HackerRank` |
| `internships_count` | integer | Number of internships completed |
| `hackathons_participated` | integer | Number of hackathons participated in |
| `freelance_experience` | integer or yes/no | `0`/`1`, or `yes`/`no` |
| `certifications_count` | integer | Number of certifications earned |
| `projects_count` | integer | Number of completed projects |
| `github_repos` | integer | Number of GitHub repositories |
| `communication_skill_score` | integer | Communication skill score |
| `teamwork_score` | integer | Teamwork score |
| `learning_consistency_score` | integer | Learning consistency score |
| `aptitude_test_score` | integer | Aptitude test score |
| `mock_interview_score` | integer | Mock interview score |
| `resume_quality_score` | integer | Resume quality score |
| `leadership_score` | integer | Leadership score |
| `extracurricular_score` | integer | Extracurricular activity score |
| `presentation_skill_score` | integer | Presentation skill score |

## Notes

- `coding_contest_skill_score` is computed automatically from `coding_contest_rating` and `coding_contest_platform`; do not send it in the request.
- `department` is encoded internally. `B.B.A` is the baseline category; other departments are mapped to one-hot columns.
- If required fields are missing or values are invalid, the API returns `400` errors.
- Ensure `model.pkl` and `scaler.pkl` remain accessible from where the server starts.
