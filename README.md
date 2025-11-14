# Salary-Prediction-using-Linear-Regression
# Salary Predictor
🧠 A Machine Learning model that predicts employee salaries based on experience, education, and job details.

# 📘 Overview
Salary Predictor is a machine learning project that uses regression analysis to estimate an employee’s salary from attributes like Age, Gender, Degree, Job Title, and Experience (years). This notebook demonstrates a complete ML workflow — including data cleaning, preprocessing, encoding, model training, and performance evaluation — built using Python and scikit-learn.

# 🧩 Key Features
✅ Load and analyze employee salary dataset ✅ Clean data by removing duplicates and missing values ✅ Encode categorical variables such as Gender, Degree, and Job Title ✅ Standardize numerical columns (Age, Experience) ✅ Train and evaluate a Linear Regression model ✅ Predict salary values and assess model performance

# ⚙️ Project Workflow
Step Description

Data Loading Load dataset using pandas (Dataset.csv).
Data Cleaning Rename columns, remove null/duplicate records.
Encoding Convert text-based columns into numeric form using LabelEncoder.
Feature Scaling Normalize numerical data with StandardScaler.
Data Splitting Split dataset (80% training, 20% testing).
Model Training Train a Linear Regression model using scikit-learn.
Evaluation Compute metrics like R², MAE, and MSE to evaluate accuracy.
# 🧠 Technologies Used
This project is developed entirely in Python 3, utilizing several powerful libraries from the data science ecosystem. Pandas and NumPy are used for data loading, manipulation, and numerical computations. Matplotlib is employed for visualizing data distributions and relationships between variables. The scikit-learn library forms the core of the machine learning pipeline, providing tools for data preprocessing, feature encoding, scaling, train-test splitting, and Linear Regression model training. Together, these technologies enable a smooth workflow from raw data cleaning to accurate salary prediction and performance evaluation.

# 📊 Model Output
After executing the notebook, you’ll obtain:

Cleaned and preprocessed dataset

Encoded and scaled features

Trained Linear Regression model

Salary predictions for test data

Evaluation results (R² Score, MAE, MSE)

# 🚀 How to Run
1️⃣ Clone the Repository git clone https://github.com/yourusername/Salary-Predictor.git

2️⃣ Navigate into the Project cd Salary-Predictor

3️⃣ Install Dependencies pip install -r requirements.txt

4️⃣ Launch the Notebook jupyter notebook Predictor.ipynb

5️⃣ Run All Cells

Execute all cells in sequence to train and test the model.

# ⚙️ Project Workflow
The workflow of this project follows a structured machine learning pipeline starting from data acquisition to model evaluation. First, the dataset is loaded using pandas from a CSV file containing employee details such as age, gender, degree, job title, experience, and salary. Next, data cleaning is performed to remove duplicate entries and handle missing values, ensuring a clean and reliable dataset. After that, the categorical variables like Gender, Degree, and Job Title are converted into numerical form using Label Encoding, allowing the machine learning model to interpret them effectively. The numerical features, such as Age and Years of Experience, are then standardized using StandardScaler to bring all values onto a comparable scale. The dataset is subsequently divided into training and testing subsets using train_test_split to fairly assess model performance. A Linear Regression model is then trained on the processed data to learn the relationship between various features and salary. Finally, the model’s accuracy is evaluated using metrics such as R² Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE) to determine its predictive performance and reliability.

# 🏁 Results & Insights
The Linear Regression model learns relationships between experience, education, and salary.

Features like Experience and Degree contribute strongly to prediction accuracy.

Model evaluation metrics (R², MAE, MSE) show how closely predictions match real-world data.
