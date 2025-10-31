# 🧠 Logistic Regression from Scratch — Diabetes Prediction

This project implements **Logistic Regression completely from scratch** in Python (without using scikit-learn).
The goal is to predict whether a person has diabetes based on health-related attributes like glucose level, BMI, and age.

---

## 📊 Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, which includes the following features:

| Feature                  | Description                      |
| ------------------------ | -------------------------------- |
| Pregnancies              | Number of times pregnant         |
| Glucose                  | Plasma glucose concentration     |
| BloodPressure            | Diastolic blood pressure (mm Hg) |
| SkinThickness            | Triceps skin fold thickness (mm) |
| Insulin                  | 2-Hour serum insulin (mu U/ml)   |
| BMI                      | Body mass index (weight/height²) |
| DiabetesPedigreeFunction | Family history function          |
| Age                      | Age in years                     |
| Outcome                  | 1 = Diabetic, 0 = Non-diabetic   |

---

## 🚀 Project Steps

### 1. **Data Preprocessing**

* Loaded dataset into a Pandas DataFrame
* Split into features (`X`) and labels (`Y`)
* Performed **train-test split**
* Applied **Z-score normalization**:
  [
  X' = \frac{X - \mu}{\sigma}
  ]
  on all features except `Outcome`

### 2. **Model Implementation**

Implemented a `LogisticRegression` class manually:

* Weight & bias initialization
* Sigmoid activation
* Forward propagation
* Gradient descent for parameter updates
* Binary cross-entropy loss
* Cost tracking during training

### 3. **Training**

Model trained using:

```python
model = LogisticRegression(learning_rate=0.01, n_iterations=2000)
model.fit(X_scaled, Y)
```

---

## 📈 Evaluation

After training, predictions were made on the test set:

```python
y_pred = model.predict(X_test_scaled)
accuracy = np.mean(y_pred == Y_test.flatten()) * 100
print(f"Accuracy: {accuracy:.2f}%")
```

* **Achieved Accuracy:** ~73%
* Cost steadily decreases with iterations, confirming proper learning.

---

## 📉 Visualization

You can visualize convergence using:

```python
plt.plot(range(0, model.n_iters, 100), model.cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Time")
plt.show()
```

---

## 🧩 File Structure

```
logistic_regression_from_scratch/
│
├── logistic_regression.py   # Implementation of LogisticRegression class
├── diabetes.csv             # Dataset
├── main.py                  # Data preprocessing, training & evaluation
└── README.md                # Project documentation
```

---

## ⚙️ Requirements

```bash
numpy
pandas
matplotlib
```

Install using:

```bash
pip install numpy pandas matplotlib
```

---

## 📚 Learnings

* Understood mathematical foundations of Logistic Regression
* Implemented gradient descent manually
* Learned importance of feature scaling and loss monitoring
* Achieved a baseline model comparable to sklearn’s performance

---

## 🏁 Next Steps

* Implement **Regularization (L2)** to prevent overfitting
* Add **learning rate scheduling** for faster convergence
* Compare results with **sklearn’s LogisticRegression**
* Extend to **multiclass classification**

---

## 👨‍💻 Author

**Satyam Mishra**
📍 Kolkata, India
🔗 [GitHub](https://github.com/satyam-mishra-dev)
🔗 [LinkedIn](https://linkedin.com/in/satyam-mishra-9329a1329)
