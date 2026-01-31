"""
train_noise_model.py
====================

This script trains a logistic‑regression classifier to predict whether there was
noise in a classroom game session based on gameplay metrics.  It reads a
CSV dataset containing labeled examples, splits the data into training and
testing sets, fits a scaler and a logistic regression model, evaluates
accuracy, and serialises the trained pipeline for later use.

Usage
-----
1. Place your dataset CSV in the same directory as this script.  By
   default, the script expects ``game_dataset.csv`` (the file you
   downloaded earlier).
2. Run the script from a terminal:

   ``python train_noise_model.py --data game_dataset.csv --model noise_model.pkl``

   You can adjust the `--data` and `--model` paths as needed.
3. After training, the script prints the model’s accuracy on a held‑out
   test set and saves the model pipeline (scaler + classifier) to the
   specified pickle file.

Predicting new data
-------------------
Once the model has been trained and saved, you can load it in another
script or an interactive session to predict the probability of noise
for new game sessions.  See the ``predict_example`` function at the
bottom of this file for a demonstration.
"""

import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def train_model(data_path: str, model_path: str) -> None:
    """Train a logistic regression model on the provided dataset and save it.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the dataset.  The CSV must include a
        column named ``noise`` representing the target label (1 = noise,
        0 = no noise).
    model_path : str
        Path to which the trained model pipeline will be saved (.pkl file).
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    if 'noise' not in df.columns:
        raise ValueError("Dataset must include a 'noise' column as the target label.")

    # Separate features and target
    X = df.drop(columns=['noise'])
    y = df['noise']

    # Split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a pipeline that standardises features then applies logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Save the trained pipeline
    joblib.dump(pipeline, model_path)
    print(f"Trained model saved to: {model_path}")


def predict_example(model_path: str, example: dict) -> float:
    """Predict the probability of noise for a single example.

    Parameters
    ----------
    model_path : str
        Path to the pickled model pipeline (as produced by ``train_model``).
    example : dict
        A dictionary of feature names and values representing a single game
        session.  The keys must match the column names in the training
        dataset except for the ``noise`` label.

    Returns
    -------
    float
        The predicted probability that the session had noise (value between
        0 and 1).
    """
    # Load the model pipeline
    model = joblib.load(model_path)

    # Construct a DataFrame from the example (model expects a 2D array)
    X_new = pd.DataFrame([example])

    # Predict the probability of the positive class (noise = 1)
    prob = model.predict_proba(X_new)[0][1]
    return prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a logistic regression model to detect noise in game sessions.")
    parser.add_argument('--data', type=str, required=True, help='Path to the CSV dataset with features and noise label.')
    parser.add_argument('--model', type=str, required=True, help='Path to save the trained model (pickle file).')
    args = parser.parse_args()

    train_model(args.data, args.model)


if __name__ == '__main__':
    main()