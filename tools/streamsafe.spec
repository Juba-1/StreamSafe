# PyInstaller spec of the Windows build: ONE folder holding two programs that share the same libraries.
#   <CLI_EXE>.exe  console program: check | prepare | run | gpu-probe  (also the engine the window starts)
#   <GUI_EXE>.exe  the window (no console): what the Start-menu shortcut opens
# One folder (not one file): no self-unpacking at every start, fewer antivirus false alarms, faster start.
# Run:  pyinstaller --noconfirm --clean tools/streamsafe.spec     (after tools/build_meta.py)
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 - SPECPATH is provided by PyInstaller
sys.path.insert(0, ROOT)

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules  # noqa: E402

from streamsafe.branding import CLI_EXE, GUI_EXE, PRODUCT_NAME  # noqa: E402

BUILD = os.path.join(ROOT, "build")

a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "tools", "win_entry.py")],
    pathex=[ROOT],
    binaries=collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("ctranslate2"),  # ctranslate2: the Arabic translation (docs/TRANSLATE.md)
    datas=[],
    hiddenimports=collect_submodules("streamsafe") + ["ctranslate2", "sentencepiece", "arabic_reshaper"],
    runtime_hooks=[os.path.join(ROOT, "tools", "pyi_rth_streamsafe.py")],
    # research / export packages are not part of the product (the product runs on ONNX Runtime)
    excludes=["torch", "torchvision", "transformers", "rfdetr", "timm", "accelerate", "onnx", "onnxscript", "pytest", "IPython", "matplotlib"],
    # onnxruntime's Python files stay on disk (not inside the archive), so that the NVIDIA build that the app
    # may unpack into the user's data folder can be put in front of them on sys.path (runtime_select.py)
    module_collection_mode={"onnxruntime": "py"},
)
pyz = PYZ(a.pure)  # noqa: F821


def program(name, console):
    return EXE(  # noqa: F821
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=name,
        console=console,
        upx=False,
        version=os.path.join(BUILD, f"version_{name}.txt"),
        icon=os.path.join(ROOT, "marketing", "icon.ico"),  # our logo on the exe, taskbar and shortcuts
    )


cli = program(CLI_EXE, True)
gui = program(GUI_EXE, False)
COLLECT(cli, gui, a.binaries, a.datas, upx=False, name=PRODUCT_NAME)  # noqa: F821
