"""First test of my first recognition model

"""
from os import listdir
import numpy as np
from tensorflow.keras.models import model_from_json
from skimage import io, transform
import matplotlib.pyplot as plt


def prediction_chiffre(nom_fichier):
    """Prediction function

    Args:
        nom_fichier (str): Number to recognize (in .png format)

    Returns:
        plus_probable (np.array): Number predicted by the model
    """
    # Lecture de l'image
    rgb = io.imread(nom_fichier)

    # Conversion en N&B
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114, 0])

    # Redimensionnement en 28x28 px
    gray28x28 = transform.resize(gray, (28, 28))

    # Filtrage du fond
    vectorized_filter = np.vectorize(lambda v: 255 if v > 128 else v)
    filtered = vectorized_filter(gray28x28)

    # Inversion des "couleurs" et normalisation des valeurs
    inverted = 255 - filtered
    reshaped = inverted.reshape(28, 28, 1) / 255.0
    batch = np.array([reshaped])

    # Prédiction
    predictions = restored_model.predict(batch)
    plus_probable = predictions.argmax(1)
    return plus_probable


with open('notebooks/First_project_image_recognition_NN2DC/model_CNN_MNIST.json', 'r') as json_file:
    loaded_model_json = json_file.read()
restored_model = model_from_json(loaded_model_json)
restored_model.load_weights(
    "notebooks/First_project_image_recognition_NN2DC/weights_CNN_MNIST.h5")

file_list = []
numbers_list = []
for file in listdir('notebooks/First_project_image_recognition_NN2DC/Images'):
    file_list.append(file)
for file in file_list:
    numbers_list.append(file[0])

TOTAL_NUMBERS = len(file_list)
validated_numbers: int = 0

for i in range(TOTAL_NUMBERS):
    number_predicted = prediction_chiffre(
        f'notebooks/First_project_image_recognition_NN2DC/Images/{file_list[i]}')
    if number_predicted[0] == int(numbers_list[i]):
        print("Bon nombre trouvé : ", number_predicted[0])
        print("File : ", file_list[i])
        print()
        validated_numbers += 1
    else:
        picture = io.imread(
            f'notebooks/First_project_image_recognition_NN2DC/Images/{file_list[i]}')

        # Conversion en N&B
        gray_picture = np.dot(picture[..., :], [0.299, 0.587, 0.114, 0])

        # Redimensionnement en 28x28 px
        gray28x28_picture = transform.resize(gray_picture, (28, 28))

        # Filtrage du fond
        vectorized_filter_for_picture = np.vectorize(
            lambda v: 255 if v > 128 else v)
        filtered_pictured = vectorized_filter_for_picture(gray28x28_picture)

        # Inversion des "couleurs" et normalisation des valeurs
        inverted_pictured = 255 - filtered_pictured
        reshaped_picture = inverted_pictured.reshape(28, 28, 1) / 255.0
        batch_false_number = np.array([reshaped_picture])

        # Prédiction
        predictions_false = restored_model.predict(batch_false_number)
        print(f"Le nombre {numbers_list[i]} n'a pas été trouvé...")
        print("File : ", file_list[i])
        print(predictions_false)
        plt.bar(range(10), predictions_false[0], tick_label=range(10))
        plt.title(f'Valeurs prédites pour le fichier {file_list[i]}')
        plt.show()
        print()
precision = 100*validated_numbers/TOTAL_NUMBERS
print('Precision : %.2f %%' % precision)
