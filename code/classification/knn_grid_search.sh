#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
values_of_k=("1 2 3 4 5 6 7 8 9 10")


# different execution modes
if [ "$1" = local ]
then
    echo "[local execution]"
    cmd="code/classification/knn.sge"
elif [ "$1" = grid ]
then
    echo "[grid execution"
    cmd="qsub code/classification/classifier.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search
# shellcheck disable=SC2128

for k in $values_of_k
do
    echo "$k"
    $cmd 'data/classification/knn_'"$k"'.pickle' --knn $k -s 7 --f1 --cohen --auc_roc
done
