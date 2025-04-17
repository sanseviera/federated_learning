import glob
import os
from collections import defaultdict
import math

# Choisis un seul bloc actif :
"""DOSSIER = "./result/defense_FedTrimmedAvg/poisoning/"
DOSSIER_OUT = "./result/defense_FedTrimmedAvg/poisoning/std"
"""

"""DOSSIER = "./result/defense_FedTrimmedAvg/label_flipping/"
DOSSIER_OUT = "./result/defense_FedTrimmedAvg/label_flipping/std"
"""

"""DOSSIER = "./result/defense_FedMedian/poisoning/"
DOSSIER_OUT = "./result/defense_FedMedian/poisoning/std"
"""

"""
DOSSIER = "./result/defense_FedMedian/label_flipping/"
DOSSIER_OUT = "./result/defense_FedMedian/label_flipping/std"
"""

DOSSIER = "./result/defense_FedMedian/non-iid/label_flipping/"
DOSSIER_OUT = "./result/defense_FedMedian/non-iid/label_flipping/std"

"""
DOSSIER = "./result/defense_FedMedian/non-iid/poisoning/"
DOSSIER_OUT = "./result/defense_FedMedian/non-iid/poisoning/std"
"""


os.makedirs(DOSSIER_OUT, exist_ok=True)
fichiers = glob.glob(os.path.join(DOSSIER, "*.txt"))

groupes = defaultdict(list)
for fichier in fichiers:
    base = os.path.basename(fichier)
    prefix = "_".join(base.split("_")[:-1])
    groupes[prefix].append(fichier)

# Calcul de l'écart-type
for prefix, fichiers_groupes in groupes.items():
    valeurs = defaultdict(list)
    autres = defaultdict(list)

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

                    valeurs[index].append(valeur)
                    autres[index].append(autre)
                except ValueError:
                    continue

    sortie_path = os.path.join(DOSSIER_OUT, f"{prefix}.txt")
    with open(sortie_path, "w") as f_out:
        for index in sorted(valeurs):
            # Écart-type (std) = sqrt( moyenne des carrés - carré de la moyenne )
            def ecart_type(liste):
                m = sum(liste) / len(liste)
                return math.sqrt(sum((x - m) ** 2 for x in liste) / len(liste))

            std_valeur = ecart_type(valeurs[index])
            std_autre = ecart_type(autres[index])
            f_out.write(f"   & {index} & {std_valeur:.10f} & {std_autre:.4f} \\\\\n")

    print(f"[OK] Écart-type généré pour: {prefix} => {prefix}.txt")


"""elif chosen_strategy == "trimmedavg":
    strategy = fl.server.strategy.FedTrimmedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        #fraction_trim=0.2,  # Tronquer 20% (10% haut, 10% bas) => total 20
        fraction_to_trim=0.2,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_fn=evaluate_function(data_split),
    )"""