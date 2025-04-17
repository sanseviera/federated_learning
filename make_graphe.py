# Choisir ici : "loss" ou "accuracy"
mode = "accuracy"

# Chemin vers le fichier d'entrée
fichier = "./result/total/model_pois_3_ecart_type.txt"

# Lecture et extraction des données
coordinates = []

with open(fichier, 'r') as f:
    for ligne in f:
        if ligne.strip() == "":
            continue
        try:
            parts = ligne.strip().split('&')
            index = int(parts[1].strip())
            loss = float(parts[2].strip())
            accuracy = float(parts[3].strip().rstrip('\\'))

            valeur = loss if mode == "loss" else accuracy
            coordinates.append((index, valeur))
        except ValueError:
            continue  # Ignore les lignes mal formées

# Détermination de la couleur en fonction du mode
color = "yellow" if mode == "loss" else "blue"

# Génération du code LaTeX
print(r"\addplot[")
print(r"    smooth, % Courbe lissée")
print(r"    mark=*,")
print(f"    color={color}")
print(r"] coordinates {")
print("    " + " ".join(f"({i},{v:.10f})" for i, v in coordinates))
print("};")
print(r"\addlegendentry{2 client honnête et 3 client malveillant}")
