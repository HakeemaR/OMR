# train_cnn_mlflow.py

import os
from pathlib import Path

import mlflow
import mlflow.tensorflow

from OMR_F import BubbleClassifierCNN   # use your existing CNN class


def main():
    here = Path(__file__).parent

    # Tell MLflow where to store experiments
    mlflow.set_experiment("OMR_CNN_Experiment")

    # Enable automatic logging: loss, accuracy, params, model graph etc.
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="cnn_synthetic_training"):
        print("\n=== TRAINING CNN WITH MLflow ENABLED ===")

        # Create the CNN (from your big script)
        cnn = BubbleClassifierCNN(input_size=(32, 32, 1))

        # You can change these if needed
        EPOCHS = 10
        TRAIN_SAMPLES = 2000
        TEST_SAMPLES = 800

        # Log hyperparameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("train_samples", TRAIN_SAMPLES)
        mlflow.log_param("test_samples", TEST_SAMPLES)

        # Train (your function already generates synthetic data)
        cnn.train_model(epochs=EPOCHS)

        # Save model locally
        model_path = here / "bubble_classifier.h5"
        cnn.save_model(str(model_path))

        # Produce PNG curves
        cnn.plot_training_curves(save_prefix="cnn_training")

        # Log plots to MLflow
        if (here / "cnn_training_accuracy.png").exists():
            mlflow.log_artifact(str(here / "cnn_training_accuracy.png"), artifact_path="plots")

        if (here / "cnn_training_loss.png").exists():
            mlflow.log_artifact(str(here / "cnn_training_loss.png"), artifact_path="plots")

        # Log the model file
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="model")

        print("\n=== MLflow run finished ===")
        print("Your metrics + plots + model are now saved in MLflow!")


if __name__ == "__main__":
    main()
