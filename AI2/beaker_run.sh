#!/bin/sh

# this script is for launching an experiment as a batch run on AI2

beaker session create --image beaker://"$(beaker account whoami --format json | jq -r .[].name)"/pytorch_dpp_1 --bare --budget ai2/reviz --gpus 2 --mount weka://ai1-default=/weka/ai1
