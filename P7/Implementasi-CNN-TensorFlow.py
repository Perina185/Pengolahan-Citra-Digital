import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data (reshape and normalize)
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert labels to categorical format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Initialize the CNN model
model = Sequential()

# Use Input layer as the first layer
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add second convolutional layer with max pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the convolutional layers
model.add(Flatten())

# Add fully connected (dense) layers
model.add(Dense(128, activation='relu'))  # Hidden layer
model.add(Dense(10, activation='softmax'))  # Output layer (for 10 classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation data
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'CNN Accuracy: {accuracy * 100:.2f}%')
