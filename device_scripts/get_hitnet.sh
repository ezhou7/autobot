#!/usr/bin/env bash
# Download the HITNet stereo model the tracking pipeline uses: PINTO0309's
# ONNX conversion of Google's HITNet trained on ETH3D (grayscale), 240x320.
# The archive holds every variant (440 MB); only one 9 MB file is kept.
#   bash device_scripts/get_hitnet.sh [destination dir, default ~/models/hitnet]
set -euo pipefail
DEST="${1:-$HOME/models/hitnet}"
URL="https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/142_HITNET/resources.tar.gz"
MEMBER="eth3d/saved_model_240x320/model_float32_opt.onnx"
SHA256="fb7ecc7d4c31e38bff93ffddbc8e98eba01d68b704b1cb9efb6c27c4c2bc660e"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fL "$URL" -o "$TMP/hitnet.tar.gz"
tar -xzf "$TMP/hitnet.tar.gz" -C "$TMP" "$MEMBER"
echo "$SHA256  $TMP/$MEMBER" | sha256sum -c -
mkdir -p "$DEST"
mv "$TMP/$MEMBER" "$DEST/eth3d_240x320.onnx"
echo "saved $DEST/eth3d_240x320.onnx"
