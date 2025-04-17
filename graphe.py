import os

#DOSSIER = "./result/defense_FedMedian/poisoning/mean/"
#DOSSIER = "./result/defense_FedMedian/label_flipping/mean/"
#DOSSIER = "./result/defense_FedTrimmedAvg/label_flipping/mean/"
#DOSSIER = "./result/defense_FedTrimmedAvg/poisoning/mean/"

#DOSSIER = "./result/defense_FedMedian/poisoning/std/"
#DOSSIER = "./result/defense_FedMedian/label_flipping/std/"
#DOSSIER = "./result/defense_FedTrimmedAvg/label_flipping/std/"
#DOSSIER = "./result/defense_FedTrimmedAvg/poisoning/std/"

#DOSSIER = "./result/defense_FedMedian/non-iid/poisoning/std/"
#DOSSIER = "./result/defense_FedMedian/non-iid/poisoning/mean/"
#DOSSIER = "./result/defense_FedMedian/non-iid/label_flipping/mean/"
DOSSIER = "./result/defense_FedMedian/non-iid/label_flipping/std/"

OUTPUT_TEX = "graphique_fedmedian.tex"

fichiers_legendes = {
    "5_honnetes.txt": "5 clients honnêtes",
    "4_honnetes_1_malveillant.txt": "4 clients honnêtes et 1 client malhonête",
    "3_honnetes_2_malveillants.txt": "3 clients honnêtes et 2 clients malhonêtes",
    "2_honnetes_3_malveillants.txt": "2 clients honnêtes et 3 clients malhonêtes",
}

data_loss = {}
data_acc = {}

for nom_fichier, legende in fichiers_legendes.items():
    path = os.path.join(DOSSIER, nom_fichier)
    losses = []
    accs = []

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

    data_loss[legende] = losses
    data_acc[legende] = accs

with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write(r"""\begin{center}
\begin{tikzpicture}
    \begin{groupplot}[
        group style={group size=2 by 1, horizontal sep=1.5cm},
        width=8cm, height=6cm,
        xlabel={$Itération$}, ylabel={$Valeur$}, grid=major,
        legend style={at={(0.5,-0.3)}, anchor=north, legend columns=1}
    ]
""")

    # Premier graphique : Loss
    f.write(r"\nextgroupplot[title={Fonction de perte (Loss)}]" + "\n")
    for couleur, (legende, valeurs) in zip(["blue", "orange", "green", "red"], data_loss.items()):
        f.write(rf"\addplot[smooth,mark=*,color={couleur}] coordinates " + "\n{\n")
        f.write("    " + " ".join(f"({i},{v:.10f})" for i, v in valeurs) + "\n};\n")
        f.write(f"\\addlegendentry{{{legende}}}\n")

    # Deuxième graphique : Accuracy
    f.write(r"\nextgroupplot[title={\text{Précision (Accuracy)}}]" + "\n")
    for couleur, (legende, valeurs) in zip(["blue", "orange", "green", "red"], data_acc.items()):
        f.write(rf"\addplot[smooth,mark=*,color={couleur}] coordinates " + "\n{\n")
        f.write("    " + " ".join(f"({i},{v:.10f})" for i, v in valeurs) + "\n};\n")
        f.write(f"\\addlegendentry{{{legende}}}\n")

    f.write(r"""\end{groupplot}
\end{tikzpicture}
\end{center}
""")

print(f"Le fichier LaTeX a été sauvegardé dans : {OUTPUT_TEX}")
