import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Indian_Student_Placement_Dataset_2025.csv"
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "gender",
    "age",
    "degree",
    "branch",
    "cgpa",
    "backlogs",
    "internships",
    "certifications",
    "coding_skills",
    "communication_skills",
    "aptitude_score",
    "projects"
]

X = df[features]

y = df["placed"]


# ==========================================
# 3. CATEGORICAL & NUMERICAL FEATURES
# ==========================================

categorical_features = [
    "gender",
    "degree",
    "branch"
]

numerical_features = [
    "age",
    "cgpa",
    "backlogs",
    "internships",
    "certifications",
    "coding_skills",
    "communication_skills",
    "aptitude_score",
    "projects"
]


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 5. CREATE ML PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ==========================================
# 8. PREDICTION
# ==========================================

y_pred = model.predict(
    X_test
)


# ==========================================
# 9. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print(
    f"Model Accuracy: {accuracy:.4f}"
)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 10. SAVE TRAINED MODEL
# ==========================================

print("\nSaving trained model...")

joblib.dump(
    model,
    MODEL_PATH
)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("===================================")

print(
    f"Saved at: {MODEL_PATH}"
)