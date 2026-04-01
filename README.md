# Student Job Prediction Server

This project is a FastAPI-based ML inference server that predicts whether a student is likely to get a job based on academic, technical, and activity-related inputs.

The model is loaded from `model.pkl`, and the API converts incoming request data into the exact feature format expected by the trained model.

## About the server

- Framework: `FastAPI`
- Model loading: `joblib`
- Data preparation: `pandas`
- Main endpoints:
  - `GET /` -> health check message
  - `POST /predict` -> prediction and confidence score

`POST /predict` returns:
- `status`: `"Got a Job"` or `"Did Not Get a Job"`
- `confidence`: probability score for the positive class

## Requirements

- Python 3.9+ recommended
- Files required in project root:
  - `app.py`
  - `model.pkl`
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

Example body (short readable keys):

```json
{
  "department": "CSE",
  "age": 22,
  "cgpa": 3.4,
  "problems_solved": 180,
  "technical_skill": 4,
  "communication_skill": 3,
  "problem_solving_ability": 3,
  "teamwork": 4,
  "extra_activities": 3,
  "projects_completed": 5,
  "internship_experience": "yes",
  "gender": "male",
  "completed_extra_courses": 4,
  "published_projects": "yes",
  "job_by_referral": "no",
  "education_aligned_job": "yes",
  "programming_languages": "Python, Java, C++",
  "activities": "Hackathons, Workshops"
}
```

Example response:

```json
{
  "status": "Got a Job",
  "confidence": 1.0
}
```

## Notes

- Prefer short keys shown above; legacy long keys are also supported for backward compatibility.
- If invalid values are sent (for example non-numeric text for numeric fields), the API returns `400` errors.
- If you move files, ensure `model.pkl` remains accessible from where the server starts.
