import numpy as np

def sigmoid_activation(x, theta):
    x = np.asarray(x) # convertit une entrée en tableau array
    theta = np.asarray(theta)
    return 1 / (1 + np.exp((-1)*np.dot(theta.T,x)))

# Utiliser une classe pour ce modèle, c'est une bonne pratique et permet de condenser le code
class NNet3:
    def __init__(self, learning_rate=0.5, maxepochs=1e4, 
                 convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)
        
    def _multiplecost(self, X, y):
        # on nourrit le réseau de neurones
        _, l2 = self._feedforward(X) # on applique la fonction feedforward
        # on calcule l'erreur
        inner = y * np.log(l2) + (1-y) * np.log(1-l2)
        # négation de l'erreur moyenne
        return -np.mean(inner)
    
    def _feedforward(self, X):
        # données de la première couche
        l1 = sigmoid_activation(X.T, self.theta0).T
        # on ajoute une colonne de 1 pour le terme de biais
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # les unités d'activation sont ensuite imputées à la couche de soprtie
        l2 = sigmoid_activation(l1.T, self.theta1)
        return l1, l2
    
    def predict(self, X):
        _, y = self._feedforward(X)
        return y
    
    def learn(self, X, y):
        nobs, ncols = X.shape
        self.theta0 = np.random.normal(0,0.01,size=(ncols,self.hidden_layer))
        self.theta1 = np.random.normal(0,0.01,size=(self.hidden_layer+1,1))
        
        self.costs = []
        cost = self._multiplecost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres+1 # fixe un coût initial à ne pas dépasser
        counter = 0 # initialise un compteur
        
        # Boucle jusqu'à la convergence
        for counter in range(self.maxepochs):
            # on nourrit le réseau
            l1, l2 = self._feedforward(X)
            
            # on démarre la backpropagation
            # Calcul des gradients
            l2_delta = (y-l2) * l2 * (1-l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1-l1)
            
            # Update des paramètres par moyenne des gradients et en multipliant 
            # par le taux d'apprentissage
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:,1:] / nobs * self.learning_rate
            
            # Stockage des coûts et vérification de la convergence
            counter += 1 # décompte
            costprev = cost # Stockage du coût actuel dans prev cost
            cost = self._multiplecost(X, y) # on obtient le nouveau coût
            self.costs.append(cost)
            if np.abs(costprev-cost) < self.convergence_thres and counter > 500:
                break