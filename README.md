
<center>  
<h1>Comparative Analysis of Machine Learning Models for Robust Skin Lesion Classification</h1>  
</center>  

# How to run the project:
1. Clone/Download the project.  
2. Go to the project folder where `manage.py` is located.  
3. Create and activate a virtual environment.  
   ```python 3.9 -m venv venv``` <br>
	Then activate env - ```venv/Scripts/activate```
5. Run this command in the terminal: `pip install -r requirements.txt`  
6. The above command will install all packages required to run the project.  
7. Run `python manage.py makemigrations` then `python manage.py migrate`  
8. Run `python manage.py runserver`  
9. Go to `http://127.0.0.1:8000/`  

# Aim  
To develop and compare different machine learning models for multi-class skin lesion classification, incorporating data augmentation and sensitivity analysis to improve robustness and generalizability.

# Objectives  
- Collect and preprocess publicly available skin lesion dataset.  
- Train and evaluate machine learning models such as Random Forest, SVM, XGBoost, CNNs, ResNet, and EfficientNet.  
- Assess the impact of data augmentation techniques (rotation, flipping, contrast adjustments, noise injection) on model robustness.  
- Perform sensitivity analysis to evaluate the effects of dataset size, augmentation intensity, and hyperparameter tuning on classification accuracy.  
- Develop a web application for real-time image classification.  

# Features:
- [x] Publicly available dataset preprocessing  
- [x] Training and evaluation of ML/DL models (Random Forest, SVM, XGBoost, CNNs, ResNet, EfficientNet)  
- [x] Data augmentation techniques implemented  
- [x] Sensitivity analysis on model robustness  
- [x] Hyperparameter tuning support  
- [x] Real-time image classification web app  
