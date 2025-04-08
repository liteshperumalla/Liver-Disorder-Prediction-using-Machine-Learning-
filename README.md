# Liver Disorder Detection using Machine Learning

## 🔍 Project Summary

This project demonstrates the use of a **Multi-Layer Perceptron (MLP)** — a type of artificial neural network — to detect liver disorders from structured clinical data. The goal is to build a predictive model that can assist in early detection and diagnosis of liver diseases using machine learning, making diagnostics more efficient and accessible.

---

## 📌 Problem Statement

Liver diseases are difficult to diagnose early with traditional methods that often require invasive procedures or expensive imaging. This project leverages supervised machine learning to create a predictive model based on patient health metrics, reducing dependency on costly diagnostics.

---

## 🧠 Model: Multi-Layer Perceptron (MLP)

We implemented an MLP classifier using Python to classify patients as having a liver disorder or not. The model is trained on labeled medical data and consists of:

- Input layer representing patient features
- Hidden layers with ReLU activation
- Output layer using sigmoid or softmax activation
- Training with backpropagation and optimization using Adam

---

## 📂 Project Structure

├── python.py # Core Python script for data preprocessing, training, and evaluation ├── dataset/ # (Recommended) Folder to store CSV data files ├── README.md # Project documentation.


> *Note: The `python.py` file includes the full model pipeline: data loading, preprocessing, model training, and evaluation.*

📈 Model Performance

Evaluation metrics: Accuracy, Precision, Recall, F1-score
Confusion Matrix and ROC curve visualization included
Trained on a structured dataset with features such as enzyme levels, age, and more
Final accuracy achieved: X% (Update based on actual model performance)

✅ Key Features

End-to-end model pipeline: preprocessing → training → testing
Customizable MLP model architecture
Model explainability using metrics and visualizations
Can be extended with additional datasets or clinical features

👨‍💻 Contributors

Perumalla Litesh
Satish Gudapati
Chittimalla Akash

📚 References

UCI Machine Learning Repository: Liver Patient Dataset
Research papers and medical resources on liver disease detection using AI
Scikit-learn, NumPy, Pandas, Matplotlib documentation

🤝 Contributions

We welcome contributions! Please fork the repository, open an issue, or submit a pull request for improvements or new features.


---

Let me know if you'd like this exported to a `.md` file or want help generating a `requirements.txt` file based on your `python.py`.

