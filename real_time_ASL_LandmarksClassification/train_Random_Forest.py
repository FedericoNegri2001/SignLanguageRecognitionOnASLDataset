from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the data with landmarks from the pickle file
data_dict = pickle.load(open('./data_with_landmarks.pickle', 'rb'))

# Convert data and labels to numpy arrays
data = np.asarray(data_dict['data'])  
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets (90% train, 10% test)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, stratify=labels)

# Initialize the Random Forest Classifier
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)   

# Print the accuracy score
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a pickle file
with open('model_rf_90.p', 'wb') as f:
    pickle.dump({'model': model}, f)
