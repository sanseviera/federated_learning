import os

# Dossiers contenant les fichiers pour les cas iid et non-iid
"""
DOSSIER_IID = "./result/defense_FedMedian/label_flipping/mean/"
DOSSIER_NON_IID = "./result/defense_FedMedian/non-iid/label_flipping/mean/"
"""
"""DOSSIER_IID = "./result/defense_FedMedian/label_flipping/std/"
DOSSIER_NON_IID = "./result/defense_FedMedian/non-iid/label_flipping/std/"
"""

"""DOSSIER_IID = "./result/defense_FedMedian/poisoning/std/"
DOSSIER_NON_IID = "./result/defense_FedMedian/non-iid/poisoning/std/"
"""

DOSSIER_IID = "./result/defense_FedMedian/poisoning/mean/"
DOSSIER_NON_IID = "./result/defense_FedMedian/non-iid/poisoning/mean/"

OUTPUT_TEX = "graphique_iid_vs_non_iid.tex"

# Fichiers à traiter avec leurs légendes
fichiers_legendes = {
    "5_honnetes.txt": "5 clients honnêtes",
    "4_honnetes_1_malveillant.txt": "4 clients honnêtes et 1 client malhonête",
    "3_honnetes_2_malveillants.txt": "3 clients honnêtes et 2 clients malhonêtes",
    "2_honnetes_3_malveillants.txt": "2 clients honnêtes et 3 clients malhonêtes",
}

# Fonction pour lire les valeurs depuis un fichier
def lire_donnees(dossier, fichier):
    path = os.path.join(dossier, fichier)
    losses, accs = [], []
    try:
        with open(path, "r") as f:
            for ligne in f:
                if ligne.strip() == "":
                    continue
                try:
                    _, index, loss, acc = ligne.strip().split("&")
                    losses.append((int(index.strip()), float(loss.strip())))
                    accs.append((int(index.strip()), float(acc.strip().rstrip("\\"))))
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Fichier introuvable : {path}")
    return losses, accs

# Lecture des données
data_loss, data_acc = {}, {}
for fichier, legende in fichiers_legendes.items():
    losses_iid, accs_iid = lire_donnees(DOSSIER_IID, fichier)
    losses_non_iid, accs_non_iid = lire_donnees(DOSSIER_NON_IID, fichier)
    data_loss[legende] = (losses_iid, losses_non_iid)
    data_acc[legende] = (accs_iid, accs_non_iid)

# Couleurs pour les courbes
couleurs = ["black", "orange", "green", "red"]
suffixes = ["", "- non\\_iid"]
styles = ["", "color=blue", "color=orange", "color=green", "color=red"]

# Génération du fichier LaTeX
with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write(r"""\begin{center}
\begin{tikzpicture}
    \begin{groupplot}[
        group style={group size=2 by 1, horizontal sep=1.5cm},
        width=8cm, height=6cm,
        xlabel={$Iteration$}, ylabel={$Valeur$}, grid=major,
        legend style={font=\scriptsize, at={(0.5,-0.3)}, anchor=north, legend columns=1}
    ]
""")

    # Première figure : Loss
    f.write(r"\nextgroupplot[title={Fonction de perte (Loss)}]" + "\n")
    for (legende, (iid, non_iid)), couleur in zip(data_loss.items(), couleurs):
        for donnees, suffixe, style in zip([iid, non_iid], suffixes, [f"color={couleur}", f"color={couleur}"]):
            f.write(rf"\addplot[smooth,mark=*,{style}] coordinates " + "{\n")
            f.write("    " + " ".join(f"({i},{v:.10f})" for i, v in donnees) + "\n};\n")
            f.write(f"\\addlegendentry{{{legende}{suffixe}}}\n")

    # Deuxième figure : Accuracy
    f.write(r"\nextgroupplot[title={\text{{Précision (Accuracy)}}}]" + "\n")
    for (legende, (iid, non_iid)), couleur in zip(data_acc.items(), couleurs):
        for donnees, suffixe, style in zip([iid, non_iid], suffixes, [f"color={couleur}", f"color={couleur}"]):
            f.write(rf"\addplot[smooth,mark=*,{style}] coordinates " + "{\n")
            f.write("    " + " ".join(f"({i},{v:.10f})" for i, v in donnees) + "\n};\n")
            f.write(f"\\addlegendentry{{{legende}{suffixe}}}\n")

    f.write(r"""\end{groupplot}
\end{tikzpicture}
\end{center}
""")

OUTPUT_TEX
print(f"Le fichier LaTeX a été sauvegardé dans : {OUTPUT_TEX}")