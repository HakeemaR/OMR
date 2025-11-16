# app.py

import os
import uuid
import json

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model_loader import bubble_model          # loads bubble_classifier.h5
from OMR_F import OMRDetector                  # your existing OMR pipeline


app = FastAPI(
    title="OMR Bubble Classification API",
    description="API for classifying bubbles and grading full OMR sheets",
    version="1.0"
)

# --- CORS so your frontend can call this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # you can restrict this later to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Helpers ------------- #

def _compute_grade(percentage: float) -> str:
    """Same grading logic you used in your console printing."""
    p = percentage
    if p >= 90:
        return "A+"
    elif p >= 85:
        return "A"
    elif p >= 80:
        return "A-"
    elif p >= 75:
        return "B+"
    elif p >= 70:
        return "B"
    elif p >= 65:
        return "B-"
    elif p >= 60:
        return "C+"
    elif p >= 55:
        return "C"
    elif p >= 50:
        return "C-"
    else:
        return "F"


# ------------- Endpoints ------------- #

@app.get("/")
def home():
    return {"message": "OMR Bubble Classifier Backend Running Successfully!"}


@app.post("/predict-bubble")
async def predict_bubble(file: UploadFile = File(...)):
    """
    Low-level endpoint: classify a single cropped bubble
    as marked / unmarked using the CNN only.
    """
    img_bytes = await file.read()
    results = bubble_model.predict(img_bytes)

    return {
        "filename": file.filename,
        "marked": results["marked"],
        "confidence": results["confidence"],
        "raw_probability": results["raw_prob"],
    }


@app.post("/grade-sheet")
async def grade_sheet(
    file: UploadFile = File(...),

    # NOTE: make both optional so we don't get 422 if one is missing.
    # Frontend can send EITHER:
    #  - answer_key_json = '{"Q1":"A","Q2":"B"}'
    #  - answers = '{"Q1":"A","Q2":"B"}'
    answer_key_json: str | None = Form(None),
    answers: str | None = Form(None),
    quiz_name: str | None = Form(None),
):
    """
    High-level endpoint:
    - Upload a FULL OMR sheet image
    - Provide the answer key either as:
        1) `answer_key_json` (stringified JSON)
        2) `answers`        (stringified JSON)
    The JSON can be:
        {"Q1":"B","Q2":"A", ...}
    or:
        {"quizName":"Math Quiz", "answers": {"Q1":"B","Q2":"A"}}
    """

    # -------- 1) Parse answer key JSON -------- #
    raw_payload = None
    if answer_key_json is not None:
        raw_payload = answer_key_json
    elif answers is not None:
        raw_payload = answers
    else:
        raise HTTPException(
            status_code=400,
            detail="No answer key provided. Send 'answer_key_json' or 'answers' as form fields."
        )

    try:
        parsed = json.loads(raw_payload)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON for answer key: {e}"
        )

    # handle both styles:
    #  1) {"Q1":"A","Q2":"B"}
    #  2) {"quizName":"Math","answers":{"Q1":"A","Q2":"B"}}
    if isinstance(parsed, dict) and "answers" in parsed:
        answer_key = parsed["answers"]
        quizName = parsed.get("quizName") or quiz_name
    else:
        answer_key = parsed
        quizName = quiz_name

    if quizName is None:
        quizName = "OMR Quiz"

    # -------- 2) Save uploaded image temporarily -------- #
    contents = await file.read()
    tmp_name = f"upload_{uuid.uuid4().hex}.jpg"
    with open(tmp_name, "wb") as f:
        f.write(contents)

    # -------- 3) Run your existing OMR detection + grading pipeline -------- #
    detector = OMRDetector(use_cnn=True, hybrid_mode=True)
    detector.marking_threshold = 0.30

    try:
        bounding_boxes, results, comparison, vis_img = detector.process_omr_with_grading(
            tmp_name, answer_key
        )
    except Exception as e:
        # clean up temp file and return error
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process OMR sheet: {e}"
        )

    # -------- 4) Save visualization image (bounding boxes, etc.) -------- #
    vis_path = f"graded_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(vis_path, vis_img)

    # -------- 5) Build summary + detailed results -------- #
    summary_answers = detector.get_answer_summary(results)  # e.g. {"Q1":"B", ...}
    percentage = float(comparison["percentage"])
    grade = _compute_grade(percentage)

    total_questions = int(comparison.get("total_questions", len(answer_key)))
    correct_list = comparison.get("correct", []) or []

    # detailed per-question result for frontend
    detailed_result = {}
    for q, correct_ans in answer_key.items():
        predicted = summary_answers.get(q)
        is_correct = q in correct_list
        detailed_result[q] = {
            "correctAnswer": correct_ans,
            "predictedAnswer": predicted,
            "isCorrect": bool(is_correct),
        }

    # optional: remove original temp file
    try:
        os.remove(tmp_name)
    except OSError:
        pass

    # -------- 6) Return response -------- #
    return {
        "filename": file.filename,
        "quizName": quizName,
        "score": comparison.get("score"),
        "total_questions": total_questions,
        "percentage": percentage,
        "grade": grade,
        "correct_questions": comparison.get("correct", []),
        "incorrect_questions": comparison.get("incorrect", []),
        "unanswered_questions": comparison.get("unanswered", []),
        "multiple_marked_questions": comparison.get("multiple_marked", []),
        "answers_detected": summary_answers,
        "visualization_image": vis_path,   # path on backend; frontend can use if needed

        # extra fields shaped nicely for the React page
        "correctCount": len(correct_list),
        "scorePercentage": percentage,
        "detailedResult": detailed_result,
    }
