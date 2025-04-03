#!/usr/bin/env bash
# File    : tinybreaker.sh
# Purpose : Build the TinyBreaker base model (prototype0)
# Author  : Martin Rizzo | <martinrizzo@gmail.com>
# Date    : Jan 19, 2025
# Repo    : https://github.com/martin-rizzo/TinyBreakerTool
# License : MIT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                              Tiny Breaker Tool
#       A set of scripts for creating and handling Tiny Breaker models.
#
#     Copyright (c) 2025 Martin Rizzo
#
#     Permission is hereby granted, free of charge, to any person obtaining
#     a copy of this software and associated documentation files (the
#     "Software"), to deal in the Software without restriction, including
#     without limitation the rights to use, copy, modify, merge, publish,
#     distribute, sublicense, and/or sell copies of the Software, and to
#     permit persons to whom the Software is furnished to do so, subject to
#     the following conditions:
#
#     The above copyright notice and this permission notice shall be
#     included in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

MODEL_TITLE="TinyBreaker prototype0"
AUTHOR="Martin Rizzo"
LICENSE="MIT"
PIXART_MODEL="PixArt-Sigma-XL-2-1024-MS.safetensors"
REFINER_MODEL="photon_v1.safetensors"


MAKETB="../maketb.sh"
MODELS_DIR=${MODELS_DIR:-"/mnt/X/DISK_X/AI_Models/PixArt"}
if [[ ! -d "$MODELS_DIR" ]]; then
    echo "Error: Directory '$MODELS_DIR' does not exist."
    echo "Please set the MODELS_DIR environment variable to a valid directory path."
    exit 1
fi
PIXART_PATH="$MODELS_DIR/$PIXART_MODEL"
REFINER_PATH="$MODELS_DIR/$REFINER_MODEL"
if [[ ! -f "$PIXART_PATH" ]]; then
    echo "Error: The model '$PIXART_MODEL' does not exist."
    echo "Please verify that the model is available in '$MODELS_DIR'"
    exit 1
fi
if [[ ! -f "$REFINER_PATH" ]]; then
    echo "Error: The model '$REFINER_MODEL' does not exist."
    echo "Please verify that the model is available in '$MODELS_DIR'"
    exit 1
fi


echo "Generating '$MODEL_TITLE' by $AUTHOR..."
"$MAKETB" --color \
  --pixart  "$PIXART_PATH" --resolution 1024 \
  --sd      "$REFINER_PATH"                  \
  --title   "$MODEL_TITLE"                   \
  --author  "$AUTHOR"                        \
  --license "$LICENSE"                       \
  --thumbnail  "prototype0_thumbnail.jpg"    \
  --metadata   "prototype0_metadata.conf"    \
  --model-type "prototype0"                  \
  -o tinybreaker_prototype0

