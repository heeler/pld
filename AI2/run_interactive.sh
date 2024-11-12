#!/bin/bash

beaker session create --image beaker://$(beaker account whoami --format json | jq -r .[].name)/pld_1 --bare --budget ai2/reviz