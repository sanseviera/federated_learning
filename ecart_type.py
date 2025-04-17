import glob
import math

# Dossier contenant tes fichiers
fichiers = glob.glob("./result/model_pois_3/*.txt")

# Dictionnaires pour stocker les sommes, carrés, et compte par index
valeurs = {}
autres = {}

# Lecture de tous les fichiers
for fichier in fichiers:
    with open(fichier, 'r') as f:
        for ligne in f:
            if ligne.strip() == "":
                continue
            try:
                _, index, valeur, autre = ligne.strip().split('&')
                index = int(index.strip())
                valeur = float(valeur.strip())
                autre = float(autre.strip().rstrip('\\'))

                # Ajout aux listes
                if index not in valeurs:
                    valeurs[index] = []
                    autres[index] = []
                valeurs[index].append(valeur)
                autres[index].append(autre)
            except ValueError:
                continue  # Ignorer les lignes mal formées

# Fonction pour calculer l'écart-type
def ecart_type(liste):
    n = len(liste)
    if n < 2:
        return 0.0
    moyenne = sum(liste) / n
    variance = sum((x - moyenne) ** 2 for x in liste) / (n - 1)
    return math.sqrt(variance)

# Affichage de l'écart-type pour chaque index
for index in sorted(valeurs):
    std_valeur = ecart_type(valeurs[index])
    std_autre = ecart_type(autres[index])
    print(f"& {index} & {std_valeur:.10f} & {std_autre:.4f} \\\\")
