#! /bin/bash 
###########################################
# Start TensorFlow Board
# Copyright (c) 2017 Hai Liang Wang. All Rights Reserved
# hailiang.hl.wang@gmail.com
# 2017-08-23
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
rootDir=$(cd $baseDir/..; pwd)
tf_board_port=6006
logdir_root=
run_prefix=run
logdir=''

# functions
function start_tensorboard(){
    tensorboard --logdir=qa:$rootDir/sdist --port $tf_board_port
}

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
start_tensorboard
