# Gender -> 1:Female 0:Male
# Churn -> 1:Yes 0:No

# What is StandardScaler?
# A tool from scikit-learn used for feature scaling.
# It standardizes numerical data so that:
# mean = 0 standard deviation = 1
# Many machine learning algorithms (like logistic regression, SVM, KNN, neural networks) work better when features are on the same scale.
# Prevents features with large values (e.g., salary in lakhs) from dominating small ones (e.g., age).

# scaler.fit_transform(X_train)
# fit() → calculates the mean and standard deviation of each column in X_train.
# transform() → uses those values to scale the data (so mean = 0, std = 1).

# joblib is a Python library used for saving and loading Python objects efficiently.
# joblib.dump(object, filename) → saves (serializes) a Python object to a file.

# Model is exported as model.pkl

# order of X: Age->Gender->Tenure->MonthlyCharges

