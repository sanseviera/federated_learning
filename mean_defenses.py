import glob
import os
from collections import defaultdict

# Dossier contenant les fichiers à traiter
"""DOSSIER = "./result/defense_FedTrimmedAvg/poisoning/"
DOSSIER_OUT = "./result/defense_FedTrimmedAvg/poisoning/mean"
"""
"""DOSSIER = "./result/defense_FedTrimmedAvg/label_flipping/"
DOSSIER_OUT = "./result/defense_FedTrimmedAvg/label_flipping/mean"
"""

"""DOSSIER = "./result/defense_FedMedian/poisoning/"
DOSSIER_OUT = "./result/defense_FedMedian/poisoning/mean"
"""

"""
DOSSIER = "./result/defense_FedMedian/label_flipping/"
DOSSIER_OUT = "./result/defense_FedMedian/label_flipping/mean"
"""

"""
DOSSIER = "./result/defense_FedMedian/non-iid/label_flipping/"
DOSSIER_OUT = "./result/defense_FedMedian/non-iid/label_flipping/mean"
"""

DOSSIER = "./result/defense_FedMedian/non-iid/poisoning/"
DOSSIER_OUT = "./result/defense_FedMedian/non-iid/poisoning/mean"

fichiers = glob.glob(os.path.join(DOSSIER, "*.txt"))

# Groupe les fichiers selon leur préfixe (sans le _1.txt, _2.txt, ...)
groupes = defaultdict(list)
for fichier in fichiers:
    base = os.path.basename(fichier)
    prefix = "_".join(base.split("_")[:-1])
    groupes[prefix].append(fichier)

# Pour chaque groupe de fichiers
for prefix, fichiers_groupes in groupes.items():
    somme_valeurs = defaultdict(float)
    somme_autres = defaultdict(float)
    compteur = defaultdict(int)

    for fichier in fichiers_groupes:
        with open(fichier, 'r') as f:
            for ligne in f:
                if ligne.strip() == "":
                    continue
                try:
                    _, index, valeur, autre = ligne.strip().split('&')
                    index = int(index.strip())
                    valeur = float(valeur.strip())
                    autre = float(autre.strip().rstrip('\\'))

                    somme_valeurs[index] += valeur
                    somme_autres[index] += autre
                    compteur[index] += 1
                except ValueError:
                    continue  # ligne mal formée

    # Écriture du résultat dans un fichier <prefix>.txt
    sortie_path = os.path.join(DOSSIER_OUT, f"{prefix}.txt")
    with open(sortie_path, "w") as f_out:
        for index in sorted(compteur):
            moyenne_valeur = somme_valeurs[index] / compteur[index]
            moyenne_autre = somme_autres[index] / compteur[index]
            f_out.write(f"   & {index} & {moyenne_valeur:.10f} & {moyenne_autre:.4f} \\\\\n")

    print(f"[OK] Moyenne générée pour: {prefix} => {prefix}.txt")