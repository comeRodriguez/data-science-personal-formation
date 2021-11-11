"""Python file for section 3 
First machine learning model 
Iris classification
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import math

iris = datasets.load_iris()
nbr_caracteristiques = 4


def sigmoid(z):
    """Activation function
    """
    return 1.0/(1 + math.exp(-z))


def predict(echantillon):
    result = 0.0
    for i in range(len(echantillon)):
        result = result + poids[i] * echantillon[i]

    result = result + offset
    return sigmoid(result)


def lost_function(expected, predicted):
    return -(expected*math.log(predicted)+(1-expected)*math.log(1-predicted))


def cost(w):
    return w**2 + w/2.0


def derivative(w):
    return 2*w + 0.5


def gradient_descent(iteration, w, learning_rate, derivative_used):
    for i in range(iteration):
        gradient = derivative_used(w)
        w = w - learning_rate*gradient
    return w, gradient


def entrainement_sur_une_iteration(echantillons, valeurs_attendues):
    pertes_totales = 0.0                # Initialise la somme des pertes à 0
    # dLw permettra de stocker les variations des pertes L par rapport aux poids w : L'(w)
    dLw = [0.0] * nbr_caracteristiques
    # dJw permettra de stocker les variations du cout par rapport aux poids w : J'(w)
    dJw = [0.0] * nbr_caracteristiques
    # dLb permettra de stocker la variation des pertes L par rapport à l'offset b : L'(b)
    dLb = 0.0
    dJb = 0.0
    # dJb permettra de stocker la variation du cout par rapport à l'offset b : J'(b)
    global offset, poids
    # m contient le nombre d'échantillons (150)
    m = len(echantillons)
    for i in range(m):
        # ech contient le i-ème échantillon : [longueur_sépale, largeur_sépale, longueur_pétale, largeur_pétale]
        ech = echantillons[i]
        # val_attendue contient la valeur attendue pour cet echantillon que doit retourner le modèle (0 ou 1)
        val_attendue = valeurs_attendues[i]
        # Appel de la fonction de prédiction pour calculer la valeur prédite par le modèle sur cet échantillon
        valeur_predite = predict(ech)
        # Additionne les pertes de chaque échantillon afin de calculer le coût
        pertes_totales = pertes_totales + \
            lost_function(val_attendue, valeur_predite)

        # Pour chaque poids, on somme la variation des pertes par rapport aux poids w : sigma[L'(w)]
        for j in range(len(poids)):
            dLw[j] = dLw[j] + ech[j]*(valeur_predite - val_attendue)

        # On somme la variation des pertes en fonction de l'offset b : sigma[L'(b)]
        dLb = dLb + (valeur_predite - val_attendue)

    # Le coût J est la valeur moyenne des pertes sur l'ensemble des 150 échantillons
    cout = pertes_totales / m
    # Calcul la variation du coût par rapport à l'offset J'(b) : Le coût est la moyenne de la variation des pertes sur l'ensemble des 150 échantillons
    dJb = dLb / m
    # Applique l'algorithme du gradient sur l'offset : bk+1 = bk - alpha*J'(b)
    offset = offset - taux_apprentissage*dJb

    for j in range(len(poids)):
        # Calcule la variation du coût par rapport au j-ème poids J'(w)
        dJw[j] = dLw[j] / m
        # Applique l'algortithme du gradient sur le j-ème poids : wk+1 = wk - alpha*J'(w)
        poids[j] = poids[j] - taux_apprentissage*dJw[j]

    return cout


poids = [0.0] * nbr_caracteristiques
offset = 0.0

taux_apprentissage = 0.1

iterations = 10000

echantillons_a_tester = iris.data
valeurs_attendues_des_echantillons = [1 if y == 2 else 0 for y in iris.target]

tableau_couts = []
for epoch in range(iterations):
    valeur_cout = entrainement_sur_une_iteration(
        echantillons_a_tester, valeurs_attendues_des_echantillons)
    tableau_couts.append(valeur_cout)

predictions = []

m = len(echantillons_a_tester)
correct = 0
for i in range(m):
    ech = echantillons_a_tester[i]
    valeur_predite = predict(ech)
    predictions.append(valeur_predite)
    if valeur_predite >= 0.6:
        valeur_predite = 1
    else:
        valeur_predite = 0
    if valeur_predite == valeurs_attendues_des_echantillons[i]:
        correct = correct + 1.0

plt.plot(range(m), predictions, label='Prévisions')
plt.plot(range(m), valeurs_attendues_des_echantillons, label='Vraies valeurs')
plt.ylabel('Prévisions')
plt.xlabel('Echantillon')
plt.legend(loc='best')
plt.title('Précision: %.2f %%' % (100 * correct/m))
plt.show()

print('Précision: %.2f %%' % (100 * correct/m))
