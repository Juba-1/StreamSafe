"""Facts of one Windows build: stamps them into the app and writes what the build tools need.

    python tools/build_meta.py [--days 30 | --expires YYYY-MM-DD] [--build N] [--out build]

  * ``streamsafe/app/build_info.py``  BUILD_DATE and EXPIRES (default: today + ``release.BETA_DAYS``): the date after
                                      which the app no longer starts a stream. Not committed; written in the build.
  * ``<out>/version_<exe>.txt``       the Windows version resource of each exe (PyInstaller ``--version-file``).
  * stdout and ``$GITHUB_OUTPUT``     version, expires, installer file name, ... as ``key=value`` lines.

The product name, publisher and exe names come from ``streamsafe/branding.py``: one place.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamsafe.app import release  # noqa: E402
from streamsafe.branding import CLI_EXE, GUI_EXE, INSTALLER_APP_ID, PRODUCT_NAME, PUBLISHER  # noqa: E402

VERSION_TEMPLATE = """# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(filevers=({a}, {b}, {c}, {d}), prodvers=({a}, {b}, {c}, {d}), mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', {company!r}),
      StringStruct('FileDescription', {description!r}),
      StringStruct('FileVersion', {version!r}),
      StringStruct('InternalName', {internal!r}),
      StringStruct('LegalCopyright', {copyright!r}),
      StringStruct('OriginalFilename', {original!r}),
      StringStruct('ProductName', {product!r}),
      StringStruct('ProductVersion', {version!r})])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def project_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def file_version(base: str, build: int) -> tuple[int, int, int, int]:
    nums = [int(n) for n in re.findall(r"\d+", base)[:3]]
    nums += [0] * (3 - len(nums))
    return nums[0], nums[1], nums[2], min(build, 65535)


def stamp_build_info(today: dt.date, expires: dt.date, version: str) -> str:
    text = (
        '"""Facts of one build, written by ``tools/build_meta.py`` in the Windows build. From source both are empty:\n'
        'a developer\'s copy has no build date and never expires."""\n\n'
        f'VERSION = "{version}"  # the version with its build number\n'
        f'BUILD_DATE = "{today.isoformat()}"  # "YYYY-MM-DD"\n'
        f'EXPIRES = "{expires.isoformat()}"  # "YYYY-MM-DD": the first day this beta build no longer starts a stream; "" = never\n'
    )
    (ROOT / "streamsafe" / "app" / "build_info.py").write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=release.BETA_DAYS, help="beta length in days from today")
    ap.add_argument("--expires", help="expiry date YYYY-MM-DD (overrides --days)")
    ap.add_argument("--build", type=int, default=0, help="build number (the 4th part of the version)")
    ap.add_argument("--out", default="build")
    args = ap.parse_args(argv)
    today = dt.date.today()
    expires = dt.date.fromisoformat(args.expires) if args.expires else today + dt.timedelta(days=args.days)
    if expires <= today:
        print("the expiry date must be in the future", file=sys.stderr)
        return 2
    base = project_version()
    a, b, c, d = file_version(base, args.build)
    version = f"{a}.{b}.{c}.{d}"
    stamp_build_info(today, expires, version)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    year = today.year
    for exe, description in ((CLI_EXE, f"{PRODUCT_NAME} engine"), (GUI_EXE, PRODUCT_NAME)):
        text = VERSION_TEMPLATE.format(
            a=a, b=b, c=c, d=d, company=PUBLISHER, description=description, version=version, internal=exe,
            copyright=f"Copyright (c) {year} {PUBLISHER}", original=f"{exe}.exe", product=PRODUCT_NAME,
        )  # fmt: skip
        (out / f"version_{exe}.txt").write_text(text, encoding="utf-8")
    facts = {
        "version": version,
        "expires": expires.isoformat(),
        "product": PRODUCT_NAME,
        "publisher": PUBLISHER,
        "cli_exe": CLI_EXE,
        "gui_exe": GUI_EXE,
        "app_guid": INSTALLER_APP_ID.strip("{}"),
        "installer": f"{PRODUCT_NAME}-Beta-Setup-{version}",
    }
    lines = [f"{k}={v}" for k, v in facts.items()]
    print("\n".join(lines))
    if gh := os.environ.get("GITHUB_OUTPUT"):
        with open(gh, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
