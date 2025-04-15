#!/bin/bash

# Par défaut
TOTAL_CLIENTS=5
NUM_ATTACKERS=1
ATTACK_TYPE="label_model_poisoning"
NUM_ROUNDS=10  # <--- Nombre d’époques (rounds) du serveur
STRATEGY="avg"  # Valeur par défaut: "avg" (FedAvg)
DATA_SPLIT="iid"  # Par défaut

# Lecture des arguments
if [ $# -ge 1 ]; then
  TOTAL_CLIENTS=$1
fi
if [ $# -ge 2 ]; then
  NUM_ATTACKERS=$2
fi
if [ $# -ge 3 ]; then
  ATTACK_TYPE=$3
fi
if [ $# -ge 4 ]; then
  NUM_ROUNDS=$4
fi
if [ $# -ge 5 ]; then
  STRATEGY=$5
fi
if [ $# -ge 6 ]; then
  DATA_SPLIT=$6
fi


echo "Lancement du serveur avec $NUM_ROUNDS rounds, type d'attaque : $ATTACK_TYPE, stratégie: $STRATEGY et data_split: $DATA_SPLIT"
python server.py --round $NUM_ROUNDS --strategy $STRATEGY &
sleep 3  # Laisse le temps au serveur de démarrer

# Lancement des clients
for ((i=0; i<$TOTAL_CLIENTS; i++)); do
  echo "Lancement du client $i"
  if [ $i -ge $(($TOTAL_CLIENTS - $NUM_ATTACKERS)) ]; then
    echo "Client $i est un attaquant ($ATTACK_TYPE)"
    python client_mal.py --node_id $i --attack_type $ATTACK_TYPE --n $TOTAL_CLIENTS --data_split $DATA_SPLIT &
  else
    python client.py --node_id $i --n $TOTAL_CLIENTS --data_split $DATA_SPLIT &
  fi
done

# Permet de tuer tous les sous-processus avec Ctrl+C
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait