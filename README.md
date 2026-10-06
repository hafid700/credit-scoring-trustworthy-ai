# 🏦 Trustworthy AI in Credit Scoring: XAI, Uncertainty & Calibration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Domain V – Trustworthy, Secure & Privacy-Preserving AI (Part 9: Explainable & Reliable AI)**
> 
> A comprehensive implementation of modern **Explainable AI (XAI)** and **Uncertainty Calibration** methods applied to financial risk assessment (Credit Scoring) using PyTorch, SHAP, LIME, and Temperature Scaling.

---

## 📌 Executive Summary

Deep learning classifiers used in critical domains like finance (Credit Scoring) often act as **black boxes** and can suffer from **overconfidence or poor probability calibration**. In regulated industries like banking, decision-making systems must be:

1. **Interpretable:** Customers and regulators need to know *why* a loan was approved or rejected (Compliance with Article 22 of GDPR).
2. **Reliable & Well-Calibrated:** Output probabilities must reflect true real-world risk rather than arbitrary confidence scores.

This project implements a complete end-to-end trustworthy AI pipeline on the **German Credit Dataset**, combining deep neural networks with cutting-edge XAI (SHAP & LIME) and post-hoc calibration techniques (Temperature Scaling & Expected Calibration Error).

---

## 🛠️ Key Capabilities & Features

* **Deep Learning Classifier (PyTorch):** Multi-Layer Perceptron (MLP) trained with BCEWithLogitsLoss, Adam optimizer, and Dropout regularization.
* **Uncertainty & Calibration Analysis:**
  * **Expected Calibration Error (ECE):** Mathematical quantification of model miscalibration.
  * **Reliability Diagrams:** Visualization of true accuracy vs. model confidence bins.
  * **Temperature Scaling:** Post-hoc optimization ($T$) to recalibrate probability distributions without degrading model accuracy.
* **Explainable AI (XAI):**
  * **Global Interpretability (SHAP):** Feature importance ranking and directional impact analysis across the entire dataset via Shapley values.
  * **Local Interpretability (LIME):** Instance-level explanations for individual credit decisions.

---

## 📊 Methodology & Technical Workflow

```
┌────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│  Data Preprocessing    │ ───► │   PyTorch Training      │ ───► │ Probability Calibration  │
│  (German Credit Data)  │      │ (Multi-Layer Perceptron)│      │  (Temperature Scaling)   │
└────────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
                                                                               │
                                                                               ▼
┌────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│ Model Deployment /     │ ◄─── │ Local Explainability    │ ◄─── │  Global Explainability   │
│ Trustworthy Scoring    │      │         (LIME)          │      │          (SHAP)          │
└────────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
```

### 1. Dataset & Preprocessing
* **Dataset:** Statlog (German Credit Data) from UCI Machine Learning Repository.
* **Task:** Binary Classification ($0 = \text{Good Credit / Approved}$, $1 = \text{Bad Credit / Risk of Default}$).
* **Key Features Included:** `duration_months`, `credit_amount`, `installment_rate`, `residence_since`, `age`, `existing_credits`, `people_liable`.
* **Data Splits:** Train (60%), Validation (20%), Test (20%) with stratified sampling and standard feature scaling.

### 2. Model Architecture
* **Type:** Multi-Layer Perceptron (MLP)
* **Layers:** 
  * Input layer: $7$ numerical features
  * Hidden Layer 1: $16$ neurons + ReLU + Dropout ($0.2$)
  * Hidden Layer 2: $8$ neurons + ReLU
  * Output Layer: $1$ neuron (Raw Logit output)
* **Optimization:** Adam Optimizer ($\text{lr} = 0.01$), Binary Cross-Entropy with Logits.

---

## 📈 Experimental Results & Key Insights

### 1. Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | `70.00%` |
| **ROC-AUC Score** | `0.6224` |

---

### 2. Model Calibration (Reliability & ECE)

Calibration measures the alignment between confidence scores and true outcomes. A well-calibrated model predicting an $80\%$ default probability should be correct exactly $80\%$ of the time.

* **Formula for Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \Big| \text{acc}(B_m) - \text{conf}(B_m) \Big|$$
* **Temperature Scaling Transformation:**
  $$\hat{p}_i = \sigma\left(\frac{z_i}{T}\right)$$

#### Calibration Comparison Table:
| Stage | ECE Score | Temperature ($T$) | Interpretation |
| :--- | :--- | :--- | :--- |
| **Uncalibrated Model** | `0.0845` | $1.000$ | Baseline PyTorch model with Dropout regularization. |
| **Calibrated Model** | `0.0890` | $0.944$ | Post-hoc recalibrated probabilities using validation set NLL. |

* **Key Takeaway:** The initial MLP model demonstrated strong calibration out-of-the-box ($\text{ECE} < 10\%$), largely due to the regularizing effect of Dropout during training.

---

### 3. Explainable AI (XAI) Analysis

#### Global Interpretability (SHAP Summary Plot)
SHAP (SHapley Additive exPlanations) ranks the global importance of features based on cooperative game theory:

1. **`duration_months`** is the most critical feature driving credit default risk. High values (Red points) consistently shift the SHAP value to the right ($\text{SHAP} > 0$), significantly increasing predicted default risk.
2. **`age`** and **`installment_rate`** follow as secondary risk drivers.
3. **`people_liable`** exhibits the lowest overall contribution to model output.

#### Local Interpretability (LIME Instance Explanation)
For an individual test case (**Client #0**):
* **Predicted Risk Probability:** `14.7%`
* **Decision:** **APPROVED (Low Risk)**
* **LIME Driving Factors:**
  * `duration_months <= -0.93` ($\text{Weight} = +0.16$ towards Good Credit)
  * `people_liable <= -0.43` ($\text{Weight} = +0.05$ towards Good Credit)
  * `credit_amount <= -0.71` ($\text{Weight} = +0.02$ towards Good Credit)

---

## 💻 Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Deep Learning Framework:** `PyTorch`
* **XAI Frameworks:** `SHAP`, `LIME`
* **Machine Learning & Data Science:** `scikit-learn`, `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 🚀 How to Run the Code

### Option 1: Run in Google Colab (Recommended)
You can directly open and execute the project notebook in Google Colab without local configuration:
1. Upload `Credit_Scoring_Trustworthy_AI.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Run all cells sequentially (`Ctrl + F9`).

### Option 2: Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/credit-scoring-trustworthy-ai.git
   cd credit-scoring-trustworthy-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch scikit-learn pandas numpy matplotlib seaborn shap lime
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

---

## 📚 Academic & Curriculum Alignment

This project directly fulfills the requirements for **Domain V – Trustworthy, Secure & Privacy-Preserving AI (Part Theme 9: Explainable & Reliable AI)**:

- [x] Comparison of Local vs. Global explanations (SHAP vs. LIME)
- [x] Evaluation of model probability calibration via Reliability Diagrams and ECE
- [x] Implementation of Temperature Scaling post-processing
- [x] Complete pipeline built with PyTorch, Scikit-Learn, and Matplotlib on CPU-friendly tabular data

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
