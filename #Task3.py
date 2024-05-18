import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv("bank.csv")
print("Initial shape of data:", data.shape)

# Check for and drop rows with any null values
data = data.dropna()
print("Shape after dropping rows with null values:", data.shape)

# Add an additional column for the target outcome
data['Predict_purchase'] = (data['y'] == 'yes').astype(int)
print(data['Predict_purchase'].head(25))

# Separate the features and target variable
y = data['Predict_purchase'].copy()
X = data.drop(['y', 'Predict_purchase'], axis=1)

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize the Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=0,max_depth=3)

# Train the model
dt.fit(X_train, y_train)

# Make predictions
dt_pred = dt.predict(X_test)

# Print the accuracy
print("Accuracy of Decision Tree Classifier: ", accuracy_score(y_test, dt_pred))
print("Precision of Decision Tree Classifier: ", precision_score(y_test, dt_pred))
print("Recall of Decision Tree Classifier: ", recall_score(y_test, dt_pred))
print("F1 Score of Decision Tree Classifier: ", f1_score(y_test, dt_pred))
# Visualize the decision tree using matplotlib
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.show()
