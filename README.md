# OMR Bubble Sheet Grading System (MLOps Coursework)

This project implements an end-to-end Optical Mark Recognition (OMR) grading system as part of an MLOps coursework. It uses:

- A **CNN model (TensorFlow/Keras)** to classify bubbles as marked / unmarked  
- An **OMR detection pipeline (OpenCV)** to find answer bubbles on the sheet  
- A **FastAPI backend** to expose a `/grade-sheet` endpoint  
- A **React frontend** that lets the user create answer keys, upload OMR sheets, and see graded results  
- (Optional) **Docker** to containerise the backend (and frontend, if required)

The goal is to demonstrate an ML system going from **model training → API → UI → deployment**.

---

## 1. Project Structure

```text
MLCW2/
├── OMR/                      # Backend (FastAPI + OMR pipeline + model)
│   ├── app.py                # FastAPI application with /grade-sheet endpoint
│   ├── OMR_F.py              # OMRDetector: OpenCV-based bubble detection + grading pipeline
│   ├── model_loader.py       # Loads bubble_classifier.h5 and wraps CNN predictions
│   ├── bubble_classifier.h5  # Trained CNN model (marked vs unmarked bubble)
│   ├── train_cnn_mlflow.py   # Training script with MLflow logging (metrics + model + plots)
│   ├── requirements.txt      # Python dependencies for backend
│   ├── Dockerfile            # Dockerfile for backend image
│   └── logs/ (optional)      # Simple JSONL logs of grading requests
│
├── omr_front/                # Frontend (React/Vite)
│   ├── frontend/             # Main React app folder (created by Vite)
│   │   ├── src/
│   │   │   ├── AnswerKeyManager.jsx   # Main UI for answer keys + uploads
│   │   │   └── main.jsx / App.jsx     # Frontend entry points
│   │   ├── package.json
│   │   └── ... (other Vite/React files)
│   └── (optional Dockerfile) # If frontend is also containerised
│
└── README.md                 # This documentation file

           +-----------------------------+
           |         User (Teacher)      |
           | - Creates answer key        |
           | - Uploads OMR sheet image   |
           +--------------+--------------+
                          |
                          v
                (React / Vite Frontend)
                          |
                          v
                 POST /grade-sheet (FastAPI)
                          |
                          v
        +-------------------------------------------+
        |        OMR Backend (FastAPI in OMR/)      |
        |                                           |
        | 1. app.py                                 |
        |    - Receives image + answer_key_json     |
        |    - Saves image temporarily              |
        |    - Calls OMRDetector                    |
        |                                           |
        | 2. OMRDetector (OMR_F.py)                 |
        |    - Preprocess with OpenCV               |
        |    - Detect circles / bubbles             |
        |    - Extract each bubble ROI              |
        |    - Use CNN (model_loader) to decide     |
        |      marked vs unmarked per option        |
        |    - Build answer summary per question    |
        |    - Compare with answer key              |
        |    - Compute score, percentage, grade     |
        |                                           |
        | 3. model_loader.py                        |
        |    - Loads bubble_classifier.h5           |
        |    - Wraps marking prediction for a ROI   |
        +-------------------------------------------+
                          |
                          v
                JSON response → Frontend
                          |
                          v
                   Result visualisation
           (Pie chart + detailed per-question list)

Conv2D(32, 3x3, relu)
MaxPooling2D(2x2)

Conv2D(64, 3x3, relu)
MaxPooling2D(2x2)

Conv2D(64, 3x3, relu)

Flatten
Dense(64, relu)
Dropout(0.5)
Dense(1, sigmoid)  # binary output (marked / unmarked)
