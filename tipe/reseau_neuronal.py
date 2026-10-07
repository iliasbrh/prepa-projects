import numpy as np

def Sigmoide(x: np.ndarray)->np.ndarray:
    return 1 / (1 + (np.exp(-x)))

def Derivee_Sigmoide(x: np.ndarray)->np.ndarray:
    # Pour des nombres trop grands en valeur absolue le calcul de exp renvoie une erreur
    # La fonction sigmoïde étant presque constante pour des valeurs très petites ou très grandes,
    # on évite les valeurs trop espacées qui seraient inutiles

    x[x < -500] = -500
    x[x > 500] = 500
    temp = np.exp(-x)
    return temp / ((1 + temp) ** 2)

class Reseau_Neuronal:
    def __init__(self, couches):
        self.couches = couches
        self.poids = []
        for i in range(len(self.couches)-1):
            self.poids.append(np.random.randn(self.couches[i], self.couches[i+1]))

        self.biais = []
        for i in couches[1:]:
            self.biais.append(np.random.randn(i))

        self.gradients = [[np.zeros(np.shape(self.poids[i])) for i in range(len(self.poids))],
                          [np.zeros(np.shape(self.biais[i])) for i in range(len(self.biais))]]

    def Reinitialiser_Gradients(self):
        # Remise des gradients à zéro
        self.gradients = [[np.zeros(np.shape(self.poids[i])) for i in range(len(self.poids))],
                          [np.zeros(np.shape(self.biais[i])) for i in range(len(self.biais))]]

    def Prediction(self, entrees):
        assert np.size(entrees) == self.couches[0], "Problème de la longueur des entrées"

        res = [np.zeros((2, j)) for j in self.couches]
        res[0][0], res[0][1] = entrees, entrees

        # res stocke l'activation des neurones avant et après l'application de la fonction sigmoïde (c'est
        # nécessaire pour le calcul des gradients ensuite)
        for i in range(len(self.couches) - 1):
            res[i+1][0] = np.matmul(res[i][1], self.poids[i]) + self.biais[i]
            res[i+1][1] = Sigmoide(res[i+1][0])

        return res

    def Calcul_Gradients(self, prediction, sorties_attendues):
        assert np.size(sorties_attendues) == self.couches[-1], "Problème de la longueur des sorties attendues"

        # Initialisation de la liste des dérivées partielles relatives à l'activation de chaque neurone
        gradients_activation = [np.zeros(j) for j in self.couches[1:]]

        # Calcul des gradients de la dernière couche avant de commencer la rétropropagation
        gradients_activation[-1] = np.multiply((2 * prediction[-1][1] - 2 * sorties_attendues), Derivee_Sigmoide(prediction[-1][0]))

        # Rétropropagation : calcul des gradients des couches précédentes
        for i in range(1, len(gradients_activation)):
            for j in range(len(gradients_activation[-i])):
                gradients_activation[-(i + 1)] += Derivee_Sigmoide(prediction[-(i + 1)][0]) * gradients_activation[-i][j] * self.poids[-i][:, j]

        # Calcul des gradients des poids
        for i in range(len(self.poids)):
            self.gradients[0][i] = prediction[i][1][:, np.newaxis] * gradients_activation[i]
        # Les biais ont la même dérivée partielle que l'activation étant donné que c'est un facteur 1 dans la dérivée partielle
        self.gradients[1] = gradients_activation

    def Entrainement(self, echantillon_apprentissage, etiquettes_apprentissage, vitesse_apprentissage=1):
        # Dans l'exemple des lettres, le tableau des etiquettes est de forme [n, 26]
        # où les n données à traiter sont les lignes et pour chaque ligne, il y a 25 zéros et un 1 :
        # par exemple pour un 'c', la ligne d'étiquette correspondante sera
        # [0, 0, 1, 0, 0, ..., (23 zéros suivent le 1)]
        n = np.shape(etiquettes_apprentissage)[0]

        for i in range(n):
            prediction = self.Prediction(echantillon_apprentissage[i])
            self.Calcul_Gradients(prediction, etiquettes_apprentissage[i])
            # Descente de gradients
            for i in range(len(self.poids)):
                self.poids[i] -= vitesse_apprentissage * self.gradients[0][i]
            for j in range(len(self.biais)):
                self.biais[j] -= vitesse_apprentissage * self.gradients[1][j]

            self.Reinitialiser_Gradients()

    def Test(self, echantillon_test, etiquettes_test):
        n = np.shape(etiquettes_test)[0]
        score = 0
        for i in range(n):
            prediction = np.argmax(self.Prediction(echantillon_test[i])[-1][1])
            if prediction == np.argmax(etiquettes_test[i]):
                score += 1
        # Renvoie la proportion de tests réussis
        return score / n
