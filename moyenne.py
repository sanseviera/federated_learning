import glob

# Dossier contenant tes fichiers (modifie le chemin selon besoin)
fichiers = glob.glob("./result/label_flip_3/*.txt")

# Dictionnaires pour stocker les sommes et le compte par index
somme_valeurs = {}
somme_autres = {}
compteur = {}

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

                # Mise à jour des sommes et du compteur
                somme_valeurs[index] = somme_valeurs.get(index, 0) + valeur
                somme_autres[index] = somme_autres.get(index, 0) + autre
                compteur[index] = compteur.get(index, 0) + 1
            except ValueError:
                continue  # Ignorer les lignes mal formées

# Affichage de la moyenne pour chaque index
for index in sorted(compteur):
    moyenne_valeur = somme_valeurs[index] / compteur[index]
    moyenne_autre = somme_autres[index] / compteur[index]
    print(f"& {index} & {moyenne_valeur:.10f} & {moyenne_autre:.4f} \\\\")
