#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
values_of_N=("100 200 500 1000")
values_of_criterion=("gini entropy")
values_of_weights=("1 0")


# different execution modes
if [ "$1" = local ]
then
    echo "[local execution]"
    cmd="code/classification/rf.sge"
elif [ "$1" = grid ]
then
    echo "[grid execution]"
    cmd="qsub code/classification/rf.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search
# shellcheck disable=SC2128

for weights in $values_of_weights
do
echo "$weights"
for criterion in $values_of_criterion
do
echo "$criterion"
for n in $values_of_N
do
    echo "$n estimators"
    $cmd 'data/classification/rf_'"$weights"'_'"$criterion"'_'"$n"'.pickle' -rf --rf_weights "$weights" --rf_criterion "$criterion" --rf_n $n -s 7 --f1 --cohen --auc_roc
done
done
done