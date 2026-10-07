import numpy as np
import matplotlib.pyplot as plt
from extraction_mots import extraction_donnees_entree, division_donnees
from time import time
import tensorflow as tf

# 330 mots par image par convention
nb_mots = 330

donnees = ['Antoine', 'Felix', 'Quentin', 'Raphael', 'Brayan']

donnees_entree = extraction_donnees_entree(donnees)
shape = donnees_entree.shape[1:]
donnees_entree = donnees_entree[:, shape[0]//2-15:shape[0]//2+15, shape[1]//2-45:shape[1]//2+45]
shape = donnees_entree.shape[1:]

for k in range(nb_mots*len(donnees)):
    donnees_entree[k] = donnees_entree[k] / np.max(donnees_entree[k])

Modele = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2700,)),
    tf.keras.layers.Dense(50, activation='sigmoid'),
    tf.keras.layers.Dense(35, activation='sigmoid'),
    tf.keras.layers.Dense(5, activation='sigmoid')])
Modele.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
res1 = []
res2 = []
t1 = time()
for i in range(50):
    donnees_entrainement, etiquettes_entrainement, donnees_test, etiquettes_test = division_donnees(donnees_entree,
                                                                                        nb_mots, 0.65, vertical=True)
    Modele.fit(donnees_entrainement, etiquettes_entrainement)
    r1 = Modele.evaluate(donnees_entrainement, etiquettes_entrainement)[1]
    r2 = Modele.evaluate(donnees_test, etiquettes_test)[1]
    res1.append(r1)
    res2.append(r2)
print(time() - t1)
plt.plot(res1)
plt.plot(res2)
plt.show()
Modele.save("Modele_pixels")


