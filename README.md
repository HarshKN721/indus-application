# indus-application

set KAGGLE_USERNAME=builduser
set KAGGLE_KEY=1234567890abcdef
pyinstaller BeamDiagnosticsSuite.spec

:: 1. Activate your environment
final_build_env\Scripts\activate

:: 2. Uninstall the pre-compiled pyinstaller
pip uninstall pyinstaller -y

:: 3. Reinstall it from source, forcing it to compile a unique bootloader locally
pip install pyinstaller --no-binary pyinstaller

pip install PyQt6 pyqtgraph matplotlib scipy numpy opencv-python-headless kaggle pyinstaller
