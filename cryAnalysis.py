import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils import resample


# Dataset Path
data_path = 'D:/NCI/Sem 3/Thesis/code_DB/dataset'

#classes
class_labels = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

# Initialize lists to hold features and labels
X = []
y = []

# Parameters for MFCC
n_mfcc = 13  # Number of MFCC features to extract
max_pad_len = 174  # Ensure all audio files have the same length

def extract_mfcc(file_path):
    """Extract MFCC features from a given audio file."""
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    # Pad or truncate the MFCCs to a fixed length
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# Loop through each class folder and extract MFCCs
for i, label in enumerate(class_labels):
    folder_path = os.path.join(data_path, label)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            mfcc = extract_mfcc(file_path)
            X.append(mfcc)
            y.append(i)  # Store the class label as an integer

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape X to be suitable for the SVM
X_flattened = X.reshape(X.shape[0], -1)  # Flatten the MFCC array

print(f'Extracted {X_flattened.shape[0]} samples with shape {X_flattened.shape}')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# Count the number of samples per class before applying SMOTE
print(f"Original class distribution: {Counter(y_train)}")

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Count the number of samples per class after oversampling
print(f"Class distribution after SMOTE: {Counter(y_train_resampled)}")

# Initialize the SVM model
svm_model = SVC(kernel='linear')

# Train the SVM model on the resampled data
svm_model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Print classification report
print(classification_report(y_test, y_pred, target_names=class_labels))