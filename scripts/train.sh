#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/..
echo "remove sdist in 3 seconds ..."
sleep 3
rm -rf sdist
set -x
python2 train.py \
    --keep_prob 0.8 \
    --lr 0.0001
