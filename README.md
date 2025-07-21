# Employee-Salary-Prediction-Abhay

## 📋 Table of Contents
1. [Problem Statement](#problem-statement)  
2. [System Development Approach](#system-development-approach)  
3. [Algorithm & Deployment](#algorithm--deployment)  
4. [Result](#result)  
5. [Conclusion](#conclusion)  
6. [How to Run the App](#how-to-run-the-app)  

---

## Problem Statement
1. The project aims to predict the salary category (<=50K or >50K) of employees based on demographic and work-related features.
2. It utilizes a machine learning classification model trained on census-like data.
3. The main objective is to automate salary prediction for HR decision support.
4. We gather and preprocess input features like age, workclass, education, occupation, etc.

---

## 🛠️ System Development Approach

- **Language & Frameworks**: Python 3.9+, Streamlit, Scikit-learn, Pandas  
- **Libraries Used**:  
  - `pandas`  
  - `numpy`  
  - `scikit-learn`  
  - `joblib`  
  - `streamlit`  
- **Methodology**:
  - Data preprocessing using label encoding for categorical variables.  
  - Model training and storage using `joblib`.  
  - Streamlit app for interactive UI and backend integration with the model.

---

## Algorithm & Deployment

1. Load dataset and explore the features.
2. Preprocess data: handle missing values, encode categorical columns using LabelEncoder.
3. Split data into train-test sets.
4. Train classification model (e.g., RandomForestClassifier, GradientBoostingClassifier).
5. Evaluate accuracy, precision, recall.
6. Save the best model using joblib.
7. Build Streamlit app for user input and display prediction.
8. Load model and encoders, transform user input, show result.

---

## Result

<img width="1061" height="695" alt="image" src="https://github.com/user-attachments/assets/4aa0c36e-ae8d-4753-98b4-3a11aa9dad28" />
<img width="1929" height="835" alt="image" src="https://github.com/user-attachments/assets/9751a4ad-7db4-4c8f-87fd-e05075e93727" />
<img width="1726" height="905" alt="image" src="https://github.com/user-attachments/assets/a9b6aa26-c8d8-490e-82ba-b588810a1ea6" />

---

## ✅ Conclusion

**1. Accurate Salary Classification**  
The model successfully predicts whether an employee's salary is above or below $50K.
It achieves high accuracy, making it valuable for HR salary assessment tasks.

**2. User-Friendly Web Interface**  
A simple web interface allows non-technical users to access predictions easily.
HR personnel can input employee details and get instant results.

**3. Challenges in Data Preprocessing**  
Handling categorical features like education and occupation was challenging.
Encoding techniques were used to make the data model-friendly.

**4. Hyperparameter Tuning and Optimization**  
Tuning model parameters was key to improving accuracy and reducing overfitting.
Techniques like grid search and cross-validation were applied.

**5. Scope for Future Improvement**  
Future work can focus on predicting salary ranges instead of just classification.
Enhancing generalization and using real-time data can boost reliability.

---

## ▶️ How to Run the App

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Amarjeetusnale/Employee-Salary-Prediction-Using-Machine-Learning-Model.git
   cd Employee-Salary-Prediction
   ```

2. **Install required libraries** :
   ```bash
   pip install pandas numpy scikit-learn joblib streamlit
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. The app will open in your default web browser at:
   ```
   http://localhost:8501
   ```

---



---

## 🙏 Thank You!
