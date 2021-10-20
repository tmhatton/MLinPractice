#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
values_of_C=("1 10 100 1000 10000 100000")
values_of_kernel=("linear poly rbf sigmoid")


# different execution modes
if [ "$1" = local ]
then
    echo "[local execution]"
    cmd="code/classification/svm.sge"
elif [ "$1" = grid ]
then
    echo "[grid execution]"
    cmd="qsub code/classification/svm.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search
# shellcheck disable=SC2128

for c in $values_of_C
do
echo "$c"
for kernel in $values_of_kernel
do
    echo "$kernel"
    $cmd 'data/classification/svm_'"$c"'_'"$kernel"'.pickle' -svm --svm_kernel "$kernel" --svm_C $c -s 7 --f1 --cohen --auc_roc
done
done
