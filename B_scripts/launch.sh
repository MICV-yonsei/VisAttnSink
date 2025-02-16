#!/bin/bash
EXP=$1
DEVICE=$2
DEBUG=$3
CMD="python src/inference.py --device $DEVICE \
--exp_config $EXP"

# Debug
[ "$DEBUG" == "debug" ] && CMD+=" debug"

echo "EXP=$EXP DEBUG=$DEBUG"
eval $CMD

: << COMMENT
bash B_scripts/launch.sh "A_exps/lv1.5_7b.yml" 0 debug
COMMENT