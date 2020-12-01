
:: Go to webnn-native directory.
cd ../
set "DAWN_DIR=%cd%"
echo "%DAWN_DIR%\out\Shared"

set PATH=%DAWN_DIR%\out\Shared;%PATH%
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
  gn gen out/Shared --ide=vs --target_cpu="x64" --args="is_component_build=true is_debug=false is_clang=false"
)

echo
ninja -C out/Shared

cd node
echo | set /p path=%DAWN_DIR% > PATH_TO_DAWN

if not exist "node_modules" (
  npm install
)

npm run all --dawnversion=0.0.1
