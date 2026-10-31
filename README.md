# 🏦 ProbabLoan – Loan Approval Prediction System

An end-to-end Machine Learning web application that predicts whether an applicant's loan request is likely to be approved or rejected based on their financial and credit profile.

Built using **Python, Scikit-Learn, and Streamlit**, ProbabLoan provides an interactive dashboard for evaluating loan eligibility through a trained classification model.

---

## 📌 Project Overview

ProbabLoan implements a supervised Machine Learning classification workflow using a **Decision Tree Classifier** to automate loan eligibility assessment.

The model evaluates key applicant attributes, including annual income, credit score, requested loan amount, employment history, and credit points, to generate instant predictions through an interactive web application.

The project demonstrates the complete ML pipeline, from data preprocessing and model training to model serialization and deployment through a Streamlit interface.

## ✨ Key Features

- **Supervised Machine Learning:** Uses Scikit-Learn's `DecisionTreeClassifier` trained on financial and credit-related indicators.
- **Interactive Web Interface:** Built with Streamlit, featuring organized input fields and a user-friendly dashboard.
- **Instant Predictions:** Generates loan approval predictions based on applicant information.
- **Pre-trained Model:** Uses Python's Pickle library to serialize and load the trained model without retraining on every application run.
- **Personalized Feedback:** Displays prediction results using the applicant's name and other entered details.
- **Input Validation:** Ensures that essential applicant information is provided before generating predictions.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Scikit-Learn | Machine Learning model development |
| Decision Tree Classifier | Loan approval classification |
| Streamlit | Interactive web application |
| Pickle | Model serialization and loading |

## 📁 Repository Structure

```text
Loan_Approval/
└── ProbabLoan/
    ├── loan_approval.csv    # Raw dataset
    ├── model.py             # Data preprocessing and model training
    ├── loan_model.pkl       # Serialized trained model
    ├── app.py               # Streamlit web application
    └── README.md            # Project documentation
```

## 📊 Dataset & Features

The project uses `loan_approval.csv`, containing applicant records with numerical and categorical attributes.

### Predictor Variables (Features)

| Feature | Data Type | Description |
|---|---|---|
| `income` | int64 | Annual income of the applicant |
| `credit_score` | int64 | Applicant's credit score (300–850) |
| `loan_amount` | int64 | Total requested loan amount |
| `years_employed` | int64 | Number of years employed |

### Metadata (Collected for UI Display)

| Feature | Data Type | Description |
|---|---|---|
| `name` | object | Applicant's name |
| `city` | object | Applicant's city |
| `points` | float64 | Evaluated credit rating/points |

### Target Variable

| Feature | Data Type | Description |
|---|---|---|
| `loan_approved` | int64 | Binary classification target (1 = Approved, 0 = Rejected) |

**Note:** Name and city are collected by the application for display purposes and are not used as model prediction features.

## 📈 Model Performance

The Decision Tree Classifier achieved an accuracy of approximately **91.5%** on the evaluation dataset.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | **93.5%** |

Accuracy is calculated using Scikit-Learn's `accuracy_score` metric by comparing the model's predicted labels against the actual loan approval outcomes.

This indicates that the model correctly classified approximately 91.5% of the samples in the evaluation dataset.

## 🚀 Getting Started

### Prerequisites

Ensure that you have the following installed:

- Python 3.9 or later
- pip (Python package manager)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/SanjanaS82/ProbabLoan.git
```

Navigate to the project directory:

```bash
cd Loan_Approval/ProbabLoan
```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn streamlit
```

### 3. Train the Machine Learning Model

To execute the preprocessing and training pipeline and generate the serialized model (`loan_model.pkl`), run:

```bash
python model.py
```

### 4. Run the Web Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Once launched, open your browser and navigate to:

```text
http://localhost:8501
```

You can now enter applicant details and generate loan approval predictions through the dashboard.

## ⚙️ Workflow Architecture

The application follows the following Machine Learning pipeline:

```text
       [ Raw Dataset ]
      loan_approval.csv
              |
              v
   [ Data Preprocessing ]
              |
              v
     [ Feature Selection ]
              |
              v
    [ Model Training ]
              |
              v
 [ Decision Tree Classifier ]
              |
              v
    [ Model Serialization ]
       loan_model.pkl
              |
              v
    [ Streamlit Application ]
              |
              v
     [ Loan Prediction ]
      Approved / Rejected
```

## 🔮 Future Enhancements

- Experiment with additional classification algorithms such as Random Forest, Logistic Regression, and Gradient Boosting.
- Implement model evaluation using accuracy, precision, recall, F1-score, and confusion matrices.
- Improve the dashboard with visual analytics and prediction summaries.
- Add explainability features to help users understand the factors influencing predictions.
- Deploy the application on a cloud platform for public accessibility.

---

**Developed as a Machine Learning project using Python and Streamlit.**