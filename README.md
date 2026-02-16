# Insurance Claim Prediction Model

## 🎯 Project Overview
This project implements a **Binary Classification** machine learning model designed to predict whether an insurance policyholder will file a claim. 
* **Target Variable:** `1` (Claim Filed) vs `0` (No Claim).

## 🧠 The Machine Learning Pipeline
Our robust, automated pipeline handles the end-to-end data journey:
1. **Data Preprocessing:** Cleans historical policy data and encodes categorical features to prepare it for the model.
2. **Feature Scaling:** Uses our `ScalerFactory` to standardize numerical features, ensuring all data points are treated equally by the algorithm.
3. **Imbalanced Data Handling:** Because insurance claims are rare events, the dataset is highly imbalanced. We utilize our `SamplingStrategyFactory` (leveraging `imbalanced-learn`) to synthesize or balance the training data, preventing the model from simply predicting "No Claim" every time.
4. **Modeling:** A configurable classifier (managed by the `ModelFactory`) learns the complex patterns that indicate a high likelihood of a future claim.


## 📊 Evaluation Metrics

Standard accuracy can be highly misleading for imbalanced datasets. Therefore, our `ModelEvaluator` heavily focuses on metrics that provide a true picture of predictive performance:
* **F1-Score:** The F1-Score is the harmonic mean of precision and recall. It provides a balanced measure of a model's performance that is especially useful for imbalanced datasets. 
* **ROC AUC:** Receiver Operating Characteristic - Area Under Curve (ROC-AUC) evaluates the model's overall performance across different thresholds. It measures the model's ability to distinguish between classes.
* **False Negative Rate (FNR):** The percentage of actual claims that our model incorrectly predicted as "No Claim." In the insurance domain, a False Negative is typically the most expensive mistake (failing to reserve capital for a valid claim), making this our primary optimization target.


## 📈 Expected Outputs
Running the pipeline (via `python src/main.py` or `./run.sh`) generates the following artifacts:
* **Trained Model File:** A serialized model ready for inference on new data.
* **Console Logs:** Real-time evaluation scores (F1, ROC AUC, FNR) outputted to your terminal.
* **Visualizations:** Automatically generated plots saved in the output directory, including:
  * *Confusion Matrix:* A visual breakdown of true vs. false predictions.
  * *ROC Curve:* A graphical representation of the model's diagnostic ability.