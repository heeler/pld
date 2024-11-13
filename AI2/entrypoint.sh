#!/bin/sh
set -e
python3.12 -m venv env
. env/bin/activate
python3.12 -m pip install --upgrade pip
pip3.12 install git+https://github.com/heeler/pld.git
printf "install done\n running"
sleep 5
cuda_check
pld_launch
echo "done"