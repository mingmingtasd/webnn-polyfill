SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WEBNN_NATIVE_DIR="$( dirname "$SCRIPT_DIR" )"
# setup openvino
OPENVINO_DIR="/opt/intel/openvino_2021"
if [ -f "$OPENVINO_DIR/bin/setupvars.sh" ]; then
  source "$OPENVINO_DIR/bin/setupvars.sh"
fi

# Go to webnn native directory to build shared library.
cd $WEBNN_NATIVE_DIR
echo "$WEBNN_NATIVE_DIR"
export LD_LIBRARY_PATH=$WEBNN_NATIVE_DIR/out/Shared${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "Syncing gclient..."
# gclient sync -D

echo "Building..."
if [ ! -f "out/Shared" ]; then
  echo "gn out/Shared"
  gn gen out/Shared --target_cpu="x64" --args="is_debug=false webnn_enable_openvino=true"
fi

echo
ninja -C out/Shared

cd node

if [ -f "node_modules" ]; then
  npm install
fi

npm run build
