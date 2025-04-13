# Projet : Attaque et Défense dans l'apprentissage fédéré
Date de rendu : 17/04/2025

Format : Rapport


## Testez directement le code
* Dans Terminal 1:
  ```bash
  > python server.py --round 10
  ```
* Dans Terminal 2:
  ```bash
  > python client.py --node_id 0 
  ```
* Dans Terminal 3:
  ```bash
  > python client_mal.py --node_id 1 
  ```
Question : Quel est le retour du client malveillant au serveur ? Comparez avec le cas où
les deux clients sont honnêtes. Quelle est votre observation sur le modèle obtenu ?


## Attaque active : 
### Attaque active :  Inversion d'étiquettes (Binôme I) 
1. Dans le fichier client_mal.py qui présente les clients malveillants,
implémentez l'attaque "inversion d'étiquettes" dans la fonction train. Par exemple, pour CIFAR10, tous les "labels" seront décalés d'un.
L'attaquant exécute cette attaque aléatoire. C'est-à-dire qu'à chaque round, il y a 50 % de probabilité que l'attaquant exécute cette attaque et 50 % de probabilité qu'il suive le protocole FedAvg.

### Attaque active :  Altération du modèle (Binôme II)
1. Dans le fichier client_mal.py qui présente les clients malveillants,
implémentez l'attaque "altération du modèle" dans la fonction train. Au lieu d'appliquer la descente de gradient,
le client appliquera une montée de gradient.
L'attaquant exécute cette attaque aléatoire. C'est-à-dire qu'à chaque round, il y a 50 % de probabilité que l'attaquant exécute cette attaque et 50 % de probabilité qu'il suive le protocole FedAvg.


**Chacun répond les questions suivantes sur leur attaque implémentée ** :
Attention : Pour chaque configuration, nous allons exécuter l’attaque cinq fois et présenter la valeur moyenne.
1. Testez votre code sur un scénario avec **cinq** clients. Augmentez le nombre de clients malveillants de 0 à 3.
Quelle est votre observation sur le modèle obtenu et pourquoi ? Affichez un graphique où l'axe des abscisses représente le nombre de clients malveillants et l'axe des ordonnées représente la précision du modèle final.

2. Retestez le scénario précédent, mais cette fois avec l'option "--data_split non_iid_class". Cette attaque est-elle plus efficace
dans cette situation et pourquoi ? Utilisez des graphiques pour montrer vos résultats.

3. Option : Vous pouvez utiliser le script **run.sh** pour lancer les clients et le serveur.
Attention, vous devez modifier le script **run.sh**  pour ajouter les clients malveillants. 

### Comparaison des performances de ces deux attaques (Ensemble)
1. Quelle attaque est plus efficace et dans quel scénario ? Pourquoi ?

## Défense 
Appliquer la défense "Médiane par coordonnées" et "Moyenne tronquée" sur le serveur,
en utilisant la stratégie fournie par flwr: class [`FedMedian`](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedmedian.py)
and class [`FedTrimmedAvg`](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedtrimmedavg.py). 

Répondez aux questions suivantes pour l'attaque d'inversion d'étiquettes (Binôme I) et pour l'attaque d'altération du modèle (Binôme II) :
N'hésitez pas à utiliser des graphiques pour montrer les résultats !

1. Quelle défense est plus efficace contre l'attaque ?
2. À partir de combien de clients malveillants la défense échoue-t-elle totalement ?
3. Comparez les cas de "--data_split iid" et "--data_split non_iid_class". La défense est-elle plus efficace
    dans quelle situation et pourquoi ?


*Remarque finale : Assurez-vous d'inclure votre nom pour l'attaque que vous avez choisi de travailler dans le rapport.*
