#~/bin/sh

# build an image and register it with beaker
# build from project root (project/AI2)

rm -rf ./artifacts/*
cp entrypoint.sh ./artifacts

beaker image create --name pytorch_dpp_1 "$(docker build -q --file Dockerfile ./artifacts --platform linux/amd64)"
