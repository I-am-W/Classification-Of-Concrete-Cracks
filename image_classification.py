import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class ImageClassifier:
    def __init__(self, positive_folder, negative_folder, noise_levels, noisy_image_folder):
        """
        Initialize the ImageClassifier with relevant folders and noise levels.
        Args:
            positive_folder (str): Folder containing positive images.
            negative_folder (str): Folder containing negative images.
            noise_levels (list): List of noise levels to generate noisy images.
            noisy_image_folder (str): Folder to store noisy images.
        """
        self.positive_folder = positive_folder
        self.negative_folder = negative_folder
        self.noise_levels = noise_levels
        self.noisy_image_folder = noisy_image_folder

    def load_image_paths_and_labels(self):
        """
        Load image paths and their corresponding labels, and store them in a DataFrame.
        Calculates width, height, and size in bytes for each image and adds these columns to the DataFrame.
        """
        positive_images = glob.glob(
            os.path.join(self.positive_folder, "*.jpg"))
        negative_images = glob.glob(
            os.path.join(self.negative_folder, "*.jpg"))

        positive_labels = ['positive'] * len(positive_images)
        negative_labels = ['negative'] * len(negative_images)

        image_paths = negative_images + positive_images
        labels = negative_labels + positive_labels

        self.df = pd.DataFrame({'image_path': image_paths, 'label': labels})
        self.df['width'] = self.df['image_path'].apply(
            lambda x: cv2.imread(x).shape[1])
        self.df['height'] = self.df['image_path'].apply(
            lambda x: cv2.imread(x).shape[0])
        self.df['size_bytes'] = self.df['image_path'].apply(
            lambda x: os.path.getsize(x))
        self.negative_df = self.df[self.df['label'] == 'negative']
        self.positive_df = self.df[self.df['label'] == 'positive']

    def plot_images_matrix(self):
        """
        Plot a matrix of sample images from both positive and negative categories.
        """

        print(f'{"*" * 50}\nApžvalginė duomenų analizė\n{"*" * 50}')
        print(
            'Atvaizduotos 5 nuotraukos su betono trūkiais ir 5 nuotraukos be betono trūkių')
        # Create scatterplots of image dimensions (width vs. height) for negative and positive categories
        negative_samples = self.negative_df .sample(5)
        positive_samples = self.positive_df.sample(5)

        # Create a subplot with 2 rows and 5 columns
        fig, axes = plt.subplots(2, 5, figsize=(12, 8))
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            if i < 5:
                img_path = negative_samples.iloc[i]['image_path']
                label = 'Negative'
            else:
                img_path = positive_samples.iloc[i - 5]['image_path']
                label = 'Positive'
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(label)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def create_image_dimensions_scatterplots(self):
        """
        Create scatterplots of image dimensions (width and height) for both categories.
        """
        print('Atvaizduotas grafikuose paveikslėlių išmatavimų pasiskirstymas. Matome, kad visų paveikslėlių dimensijos yra 277 x 277')
        # Create two scatter plots side by side
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.scatterplot(data=self.negative_df, x='width',
                        y='height', label='Negative', alpha=0.7)
        plt.title('Image Dimensions for Negative Category')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels')

        plt.subplot(1, 2, 2)
        sns.scatterplot(data=self.positive_df, x='width',
                        y='height', label='Positive', alpha=0.7)
        plt.title('Image Dimensions for Positive Category')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels')

        plt.tight_layout()
        plt.show()

    def create_image_size_histograms(self):
        """
        Create histograms of image sizes for both categories.
        """
        print('Paveikslėlių su trūkiais ir be trūkių užimamos vietos pasiskirstymas')
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(self.negative_df['size_bytes'],
                     kde=True, label='Negative')
        plt.title('Image Size Histogram for Negative Category')
        plt.xlabel('Size (bytes)')
        plt.ylabel('Count')

        plt.subplot(1, 2, 2)
        sns.histplot(self.positive_df['size_bytes'],
                     kde=True, label='Positive')
        plt.title('Image Size Histogram for Positive Category')
        plt.xlabel('Size (bytes)')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()
        print('Paveikslėlių be trūkių užimamos vietos statistika:')
        print(self.negative_df['size_bytes'].describe())
        print(
            f'Asimetrijos: koeficientas: {self.negative_df["size_bytes"].skew()}')
        print(f'Ekscesas: {self.negative_df["size_bytes"].kurtosis()}\n')

        print('Paveikslėlių su trūkiais užimamos vietos statistika:')
        print(self.positive_df['size_bytes'].describe())
        print(
            f'Asimetrijos koeficientas: {self.positive_df["size_bytes"].skew()}')
        print(f'Ekscesas: {self.positive_df["size_bytes"].kurtosis()}')

        print('Galime matyti, kad paveisklėlių su trūkiais vidutinė užimamos vietos reikšmė yra didesnė. Taip pat asimetrijos koeficientas ir ekscesas yra mažesnis\n')

    def plot_data_distribution(self):
        """
        Plot the distribution of negative and positive images.
        """
        print('Duomenų rinkinį sudaro 20000 nuotraukų su betono trūkiais ir 20000 be betono trūkių.\n')
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x='label')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Distribution of Negative and Positive Images')
        plt.show()

    def split_train_test_data(self):
        """
        Split the data into training and testing sets.
        """
        print(f'{"*" * 50}\nModelio sudarymas\n{"*" * 50}')
        train_df, test_df = train_test_split(self.df.sample(10000, random_state=1),
                                             train_size=0.80,
                                             shuffle=True,
                                             random_state=1
                                             )
        self.train_df = train_df
        self.test_df = test_df

    def create_data_generators(self):
        """
        Create data generators for training, validation, and testing.
        """
        # Create a training data generator with rescaling and a 25% validation split.
        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=0.25)

        # Create a training data generator from a DataFrame for the training subset.
        self.train_data = train_gen.flow_from_dataframe(self.train_df,
                                                        x_col='image_path',
                                                        y_col='label',
                                                        target_size=(120, 120),
                                                        color_mode='rgb',
                                                        class_mode='binary',
                                                        batch_size=32,
                                                        shuffle=True,
                                                        seed=42,
                                                        subset='training'
                                                        )

        # Create a validation data generator from the same DataFrame for the validation subset.
        self.val_data = train_gen.flow_from_dataframe(self.train_df,
                                                      x_col='image_path',
                                                      y_col='label',
                                                      target_size=(120, 120),
                                                      color_mode='rgb',
                                                      class_mode='binary',
                                                      batch_size=32,
                                                      shuffle=True,
                                                      seed=42,
                                                      subset='validation'
                                                      )

        # Create a test data generator from a different DataFrame for testing.
        self.test_data = train_gen.flow_from_dataframe(self.test_df,
                                                       x_col='image_path',
                                                       y_col='label',
                                                       target_size=(120, 120),
                                                       color_mode='rgb',
                                                       class_mode='binary',
                                                       batch_size=32,
                                                       shuffle=False,
                                                       seed=42
                                                       )
        # Store the training data generator in the class for later use.
        self.train_gen = train_gen

    def build_model(self):
        """
        Build a convolutional neural network model.
        """
        # Define the input layer with a shape of (120, 120, 3)
        inputs = tf.keras.Input(shape=(120, 120, 3))

        # Add a 2D convolutional layer with 16 filters and a 3x3 kernel, using ReLU activation
        x = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3), activation='relu')(inputs)

        # Add a max-pooling layer with a 2x2 pool size
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        # Add another 2D convolutional layer with 32 filters and a 3x3 kernel, using ReLU activation
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3), activation='relu')(x)

        # Add another max-pooling layer with a 2x2 pool size
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        # Add a global average pooling layer
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Add a dense (fully connected) layer with 1 output unit and a sigmoid activation
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        # Create the model with the specified input and output layers
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model with the Adam optimizer, binary cross-entropy loss, and accuracy metric
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self):
        """
        Train the neural network model.
        """
        print('Apmokomas modelis')
        history = self.model.fit(
            # Train the model using training data.
            self.train_data,
            # Use validation data to monitor the model's performance during training.
            validation_data=self.val_data,
            # Set the number of training epochs to 69.
            epochs=69,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    # Monitor the validation loss to decide when to stop training.
                    monitor='val_loss',
                    # Allow up to 3 consecutive epochs without improvement before stopping.
                    patience=3,
                    # Restore the model's weights to the best ones when training stops.
                    restore_best_weights=True
                )
            ]
        )
        print('Modeliui sudaryti užteko 20 epochų,  kur treniravimo ir patvirtinimo praradimas (loss) siekė apie 0.06')
        # Call a method 'plot_training_history' to visualize the training history.
        self.plot_training_history(history)
        print('Modeliui sudaryti užteko 20 epochų,  kur treniravimo ir patvirtinimo praradimas (loss) siekė apie 0.06')
        print(f'{"*" * 50}\nModelio vertinimas\n{"*" * 50}\n')
        print('Įvertinus modelį nustatyta, kad modelis gali nustatyti betono įtrūkimus 97% tikslumu')

    def plot_training_history(self, history):
        """
        Plot the training and validation loss over epochs.
        """
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, label='Training Loss', marker='o')
        plt.plot(epochs, val_loss, label='Validation Loss', marker='o')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.show()

    def evaluate_model(self, test_data):
        """
        Evaluate the model on the test data and display the results.
        """

        # Evaluate the model using the provided test data.
        test_results = self.model.evaluate(test_data)

        # Extract the test loss and test accuracy from the evaluation results.
        test_loss = test_results[0]
        test_accuracy = test_results[1]

        # Print the test loss and test accuracy.
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        # Make predictions using the model on the test data and convert them to 0 or 1 based on a threshold of 0.5.
        y_pred = np.squeeze((self.model.predict(test_data) >= 0.5).astype(int))

        # Calculate the confusion matrix to assess the model's performance.
        confusion_mat = confusion_matrix(test_data.labels, y_pred)

        # Create a heatmap to visualize the confusion matrix.
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])

        # Label the x and y axes and provide a title for the heatmap.
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

        # Generate a classification report and print it
        class_report = classification_report(
            test_data.labels, y_pred, target_names=["Negative", "Positive"])
        print("Classification Report:\n", class_report)

    def add_noise_to_image(self, image_path, noise_level):
        """
        Add noise to an image with a specified noise level.
        Args:
            image_path (str): Path to the image.
            noise_level (float): Level of noise to add.
        Returns:
             The noisy image.
        """
        img = cv2.imread(image_path)
        # Generate random noise with a normal distribution
        noise = np.random.normal(0, noise_level * 255,
                                 img.shape).astype(np.uint8)
        # Add the generated noise to the image
        noisy_image = cv2.add(img, noise)
        return noisy_image

    def generate_noisy_images(self):
        """
        Generate noisy images with different noise levels and evaluate the model on them.
        """
        print('Paveikus testinius paveiklėlius vis didesniais (0.1, 0.420 ir 0.6) triukšmo lygiais, modelio tikslumas vis labiau blogėjo.')
        # Loop through different noise levels
        for level in self.noise_levels:
            test2_df = pd.DataFrame(columns=['image_path', 'label'])
            folder_path = os.path.join(self.noisy_image_folder, str(level))

            # If the folder for the current noise level already exists, remove its contents
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
            else:
                os.makedirs(folder_path)

            # Loop through rows in the test_df DataFrame
            for index, row in self.test_df.iterrows():
                image_path = row['image_path']
                label = row['label']

                # Add noise to the image and save it in the noisy_image_folder
                noisy_image = self.add_noise_to_image(image_path, level)
                filename_with_extension = os.path.basename(image_path)
                new_path = os.path.join(folder_path, filename_with_extension)
                new_row = {'image_path': new_path, 'label': label}

                # Concatenate the new row to the test2_df DataFrame
                test2_df = pd.concat(
                    [test2_df, pd.DataFrame([new_row])], ignore_index=True)

                # Save the noisy image to the specified path
                cv2.imwrite(new_path, noisy_image)

            # Create a data generator from the modified DataFrame
            test2_data = self.train_gen.flow_from_dataframe(test2_df,
                                                            x_col='image_path',
                                                            y_col='label',
                                                            target_size=(
                                                                120, 120),
                                                            color_mode='rgb',
                                                            class_mode='binary',
                                                            batch_size=32,
                                                            shuffle=False,
                                                            seed=42
                                                            )
            self.evaluate_model(test2_data)

    def run_classification(self):
        """
        Run the entire image classification process.
        """
        self.load_image_paths_and_labels()
        self.plot_images_matrix()
        self.create_image_dimensions_scatterplots()
        self.create_image_size_histograms()
        self.plot_data_distribution()
        self.split_train_test_data()
        self.create_data_generators()
        self.build_model()
        self.train_model()
        self.evaluate_model(self.test_data)
        self.generate_noisy_images()
