"""The product's name, in ONE place. The name may still change: the window, the installer, the Start-menu
shortcut, the per-user data folder and every message a streamer reads take it from here.

``tools/build_meta.py`` hands it to the Windows build (exe version resource, Inno Setup script)."""

PRODUCT_NAME = "StreamSafe"
PUBLISHER = "StreamSafe"
CLI_EXE = PRODUCT_NAME  # <name>.exe: the console program (check | prepare | run), also the engine the window starts
GUI_EXE = f"{PRODUCT_NAME}App"  # <name>App.exe: the window (no console), what the Start-menu shortcut opens
# Windows identifies an installed program by this id, not by its name: it must NOT change when the product is
# renamed, or an update would install a second copy beside the old one
INSTALLER_APP_ID = "{6F0C4B0E-5B0A-4E0B-9D53-5E1B7E2C8A41}"
