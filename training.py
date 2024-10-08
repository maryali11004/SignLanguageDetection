import os
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from preprocessing import preprocess_data  # Import the preprocessing function

# Load preprocessed data
X_train, X_test, y_train, y_test = preprocess_data()

# Setup TensorBoard callback
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))  # Adjust input shape to match keypoints
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Adjust output to match action count

# Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train model
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# Make predictions
yhat = model.predict(X_test)

# Save model weights
model.save('action.h5')

# Evaluation
ytrue = np.argmax(y_test, axis=1)
yhat = np.argmax(yhat, axis=1)

# Confusion matrix and accuracy
conf_matrix = multilabel_confusion_matrix(ytrue, yhat)
accuracy = accuracy_score(ytrue, yhat)

# Print results
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy Score: {accuracy}")
