# music_genre_classifier/train.py
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.models import Sequential
from keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import regularizers # Import regularizers

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
SAMPLE_RATE = 22050
FIXED_TIME_FRAMES = 200 # Approx 4.6 seconds
DURATION_SECONDS = 30 # GTZAN files are 30 seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION_SECONDS
SAMPLES_PER_SEGMENT = (FIXED_TIME_FRAMES -1) * 512 + 1024 # Calculate samples needed for FIXED_TIME_FRAMES spectrogram frames

# --- Data Augmentation Function ---
def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_y = y + noise_factor * noise
    # Cast back to same data type
    augmented_y = augmented_y.astype(type(y[0]))
    return augmented_y

def extract_features_librosa(file_path, augment=False, segment_start_time=0):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Ensure audio is long enough for the segment
        start_sample = int(segment_start_time * sr)
        if len(y) < start_sample + SAMPLES_PER_SEGMENT:
            # If not, pad the original audio or use from start
            # For simplicity here, we'll just take from start if segment_start_time is too large
            if len(y) < SAMPLES_PER_SEGMENT: # If entire audio is too short (shouldn't happen with GTZAN)
                 y_segment = np.pad(y, (0, SAMPLES_PER_SEGMENT - len(y)), mode='constant')
            else:
                y_segment = y[:SAMPLES_PER_SEGMENT]
        else:
            y_segment = y[start_sample : start_sample + SAMPLES_PER_SEGMENT]


        if augment:
            # Example: Add noise
            y_segment = add_noise(y_segment, noise_factor=np.random.uniform(0.001, 0.005))
            # Other augmentations like pitch shift or time stretch could be added here
            # pitch_shift_steps = np.random.randint(-2, 3) # -2 to +2 semitones
            # if pitch_shift_steps != 0:
            #    y_segment = librosa.effects.pitch_shift(y_segment, sr=sr, n_steps=pitch_shift_steps)


        mel = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max) # mel_db shape is (n_mels, time_frames)

        # Padding/truncating spectrogram columns (time_frames)
        if mel_db.shape[1] < FIXED_TIME_FRAMES:
            pad_width = FIXED_TIME_FRAMES - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :FIXED_TIME_FRAMES]

        return mel_db
    except Exception as e:
        print(f"Error processing {file_path} at segment starting {segment_start_time}s: {e}")
        return None

def load_data(data_dir, augment_training_data=False, num_segments_per_track=1):
    features, labels = [], []
    max_start_time = DURATION_SECONDS - (SAMPLES_PER_SEGMENT / SAMPLE_RATE) # Max start time for a segment

    for genre_idx, genre in enumerate(GENRES):
        genre_dir = os.path.join(data_dir, genre)
        for file_name in os.listdir(genre_dir):
            if file_name.endswith('.wav'):
                path = os.path.join(genre_dir, file_name)

                # For training data, extract multiple (augmented) segments
                # For validation/test data (if handled separately), usually extract one central segment without augmentation
                # Here, we control augmentation via augment_training_data for simplicity
                # but a more robust pipeline would split data first, then augment only training.

                for i in range(num_segments_per_track):
                    # For multiple segments, pick random start times
                    if num_segments_per_track > 1:
                        segment_start = np.random.uniform(0, max_start_time)
                    else:
                        segment_start = 0 # Default to start of track if only one segment

                    # Apply augmentation only if specified (typically for training data)
                    feature = extract_features_librosa(path, augment=augment_training_data, segment_start_time=segment_start)
                    if feature is not None:
                        features.append(feature)
                        labels.append(genre_idx)

    return np.array(features), np.array(labels)


from keras import regularizers # Ensure this is imported

def build_model(input_shape, num_classes):
    l2_reg = regularizers.l2(0.001) # Define the regularizer once

    model = Sequential([
        Input(shape=input_shape),

        SeparableConv2D(32, (3, 3), activation='relu', padding='same',
                        depthwise_regularizer=l2_reg, # Apply to depthwise kernel
                        pointwise_regularizer=l2_reg),# Apply to pointwise kernel
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        SeparableConv2D(64, (3, 3), activation='relu', padding='same',
                        depthwise_regularizer=l2_reg,
                        pointwise_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        SeparableConv2D(128, (3, 3), activation='relu', padding='same',
                        depthwise_regularizer=l2_reg,
                        pointwise_regularizer=l2_reg),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2_reg), # Dense layer accepts kernel_regularizer
        BatchNormalization(),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test_one_hot):
    preds_proba = model.predict(X_test)
    y_pred_indices = np.argmax(preds_proba, axis=1)
    y_true_indices = np.argmax(y_test_one_hot, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_indices, y_pred_indices, target_names=GENRES, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=GENRES, yticklabels=GENRES, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    data_dir = 'Data/genres_original'
    # Load data:
    # - Augment training data by extracting multiple random segments with noise
    # - This effectively increases the size and diversity of the training set.
    print("Loading and preprocessing data...")
    # To properly apply augmentation only to training, you'd split filenames first.
    # For a simpler approach here, we load all data with potential for augmentation,
    # then split. A more robust pipeline would handle this more cleanly.
    # We will load with num_segments_per_track=1 and augment=False for initial split,
    # then can consider a more complex augmentation strategy for X_train later if needed.

    # Simpler initial loading:
    X_all, y_all = load_data(data_dir, augment_training_data=False, num_segments_per_track=1)
    
    # Reshape features for CNN (add channel dimension)
    X_all = X_all[..., np.newaxis]
    # Convert labels to categorical
    y_all_one_hot = to_categorical(y_all, num_classes=len(GENRES))

    # Split into training and testing sets
    X_train, X_test, y_train_one_hot, y_test_one_hot = train_test_split(
        X_all, y_all_one_hot, test_size=0.25, random_state=42, stratify=y_all # Stratify by labels
    )
    
    print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train_one_hot.shape}")
    print(f"Test set shape: {X_test.shape}, Test labels shape: {y_test_one_hot.shape}")

    # --- Augment Training Data (Optional - more advanced) ---
    # If you want to augment ONLY the training split after splitting, you could reload or process X_train paths.
    # For now, we'll proceed without post-split augmentation to keep it simpler.
    # The `augment_training_data` and `num_segments_per_track` in `load_data` provide
    # a basic way to increase data if applied before splitting, but it mixes augmented
    # data across potential train/val/test splits if not handled carefully.
    
    # Build the model
    input_shape_val = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (n_mels, time_frames, channels)
    model = build_model(input_shape_val, num_classes=len(GENRES))
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1), # Increased patience
        ModelCheckpoint('music_genre_classifier.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1) # Learning rate scheduler
    ]

    # Train the model
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train_one_hot,
        epochs=100, # Increased epochs, EarlyStopping will handle stopping
        batch_size=32, # Tried smaller batch size
        validation_data=(X_test, y_test_one_hot),
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plot_history(history)

    # Evaluate the model (it will use the best weights restored by EarlyStopping)
    print("\nEvaluating model on the test set...")
    evaluate_model(model, X_test, y_test_one_hot)

    # To load the best model explicitly if needed elsewhere:
    # from keras.models import load_model
    # best_model = load_model('music_genre_classifier.keras')
    # evaluate_model(best_model, X_test, y_test_one_hot)

if __name__ == '__main__':
    main()