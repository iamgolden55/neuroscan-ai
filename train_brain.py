import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. PREPARE THE IMAGES
# We use a generator to load images on the fly (saves memory)
# We also "augment" the data (zoom, flip) to make the AI smarter
datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel colors (0-255 -> 0-1)
    validation_split=0.2   # Use 20% of images to test the AI
)

# Load Training Data (80%)
train_generator = datagen.flow_from_directory(
    'brain_tumor_dataset',  # FOLDER NAME (Make sure this matches your download!)
    target_size=(150, 150), # Resize all images to 150x150
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Load Validation Data (20%)
validation_generator = datagen.flow_from_directory(
    'brain_tumor_dataset',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 2. BUILD THE CNN (The Deep Learning Brain)
model = Sequential()

# Layer 1: The "Edges" Eye
# Finds simple lines and borders
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))

# Layer 2: The "Shapes" Eye
# Finds circles (tumor shapes) and curves (skull)
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Layer 3: The "Texture" Eye
# Finds complex patterns inside the brain tissue
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten: Turn the 2D image map into a 1D list of numbers
model.add(Flatten())

# Layer 4: The Decision Maker (Neurons)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5)) # Randomly turn off 50% neurons to prevent memorization (Overfitting)
model.add(Dense(1, activation='sigmoid')) # Output: 0 (No) or 1 (Yes)

# 3. COMPILE & TRAIN
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Starting Training... (This usually takes 5-10 mins)")
history = model.fit(
    train_generator,
    epochs=10,  # Go through the dataset 10 times
    validation_data=validation_generator
)

# 4. SAVE THE BRAIN
model.save('brain_tumor_model.h5')
print("Model Saved Successfully as 'brain_tumor_model.h5'")