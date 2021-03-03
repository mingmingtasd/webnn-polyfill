
:: Go to webnn-native directory.
cd ../
set "WEBNN_NATIVE_DIR=%cd%"
echo "%WEBNN_NATIVE_DIR%\out\Shared"

set PATH=%WEBNN_NATIVE_DIR%\out\Shared;%PATH%
::setup openvino
set "OPENVINO_DIR=%ProgramFiles(x86)%\IntelSWTools\openvino"
if exist "%OPENVINO_DIR%\bin\setupvars.bat" (
  call "%OPENVINO_DIR%\bin\setupvars.bat"
)

echo
echo "Syncing gclient..."
::gclient sync -D

echo "Building..."
if not exist "out/Shared" (
  gn gen out/Shared --ide=vs --target_cpu="x64" --args="is_component_build=true is_debug=false is_clang=false webnn_enable_openvino=true"
)

echo
ninja -C out/Shared

cd node
::echo | set /p path=%WEBNN_NATIVE_DIR% > PATH_TO_WEBNN_NATIVE

if not exist "node_modules" (
  npm install
)

npm run build
