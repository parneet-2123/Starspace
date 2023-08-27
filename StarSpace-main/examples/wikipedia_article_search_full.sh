#!/usr/bin/env bash
#
# Copyright (c) funcoding, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET=(
  wikipedia
)

MODELDIR=/tmp/spacexplore/models
DATADIR=/tmp/spacexplore/data

mkdir -p "${MODELDIR}"
mkdir -p "${DATADIR}"

if [ ! -f "${DATADIR}/${DATASET[i]}_shuf_train5M.txt" ]
then
    echo "Downloading wikipedia train data"
    wget -c "https://dl.fbaipublicfiles.com/spacexplore/wikipedia_train5M.tgz" -O "${DATADIR}/${DATASET[0]}_train.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[0]}_train.tar.gz" -C "${DATADIR}"
  fi

echo "Compiling spacexplore"

make

echo "Start to train on wikipedia data (meant to replicate experiment from paper, this will take a while to train):"

./spacexplore train \
  -trainFile "${DATADIR}"/wikipedia_shuf_train5M.txt \
  -model "${MODELDIR}"/wikipedia_article_search_full \
  -trainMode 2 \
  -initRandSd 0.01 \
  -adagrad true \
  -ngrams 1 \
  -lr 0.05 \
  -margin 0.05 \
  -epoch 20 \
  -thread 40 \
  -dim 300 \
  -negSearchLimit 100 \
  -dropoutRHS 0.8 \
  -fileFormat labelDoc \
  -similarity "cosine" \
  -minCount 5 \
  -normalizeText true \
  -verbose true

if [ ! -f "${DATADIR}/${DATASET[i]}_test10k.txt" ]
then
    echo "Downloading wikipedia test data"
    wget -c "https://dl.fbaipublicfiles.com/spacexplore/wikipedia_devtst.tgz" -O "${DATADIR}/${DATASET[0]}_test.tar.gz"
    tar -xzvf "${DATADIR}/${DATASET[0]}_test.tar.gz" -C "${DATADIR}"
fi

echo "Start to evaluate trained model:"

./spacexplore test \
  -testFile "${DATADIR}"/wikipedia_test10k.txt \
  -basedoc "${DATADIR}"/wikipedia_test_basedocs.txt \
  -model "${MODELDIR}"/wikipedia_article_search_full \
  -thread 20 \
  -trainMode 2 \
  -normalizeText true \
  -verbose true
