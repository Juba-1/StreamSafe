"""Release settings: THE one place for every address the app may contact and every pinned download.

The app makes no network call that is not listed here (docs/SECURITY_POSTURE.md):
  * the model files (``MODEL_BASE_URL``) and, when set, the signed update manifest (``UPDATE_MANIFEST_URL``),
  * NVIDIA's libraries for the fast NVIDIA mode (``NVIDIA_WHEELS``), only after the streamer says yes,
  * the streamer's own stream output.
Everything downloaded is data or a vendor library pinned by sha256 here or in the signed manifest; nothing
downloaded is ever run as a program by us.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- models ---------------------------------------------------------------------------------------------------
# A folder of files with a ``manifest.json`` beside them (name -> bytes + sha256 of every file, written by
# ``tools/export_onnx.py``). The app downloads the manifest, then every file it lists, and checks each file against
# the manifest. The repository may still be private (then Hugging Face answers 401 / 404: the app says so and
# offers "choose a folder or zip we sent you"). The risk heads are NOT here: they ship inside the installer.
MODEL_BASE_URL = "https://huggingface.co/juba-1/streamsafe-models/resolve/main/v1/"

# --- live Arabic translation of on-screen text (docs/TRANSLATE.md) ------------------------------------------------
# Downloaded the first time the streamer switches it on (not part of the installer, not part of the model pack):
# OCR models, the opus-mt-en-ar translator and two fonts, with their licence texts. Every file is pinned by size and
# sha256 (``tools/build_translate_pack.py`` prints this table); nothing that does not match is kept.
TRANSLATE_BASE_URL = MODEL_BASE_URL + "translate/"
TRANSLATE_DOWNLOAD_MB = 96  # shown to the streamer before the download
TRANSLATE_FILES: dict[str, tuple[str, int]] = {
    "LICENSE-Apache-2.0.txt": ("cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30", 11358),
    "LICENSE-CC-BY-4.0.txt": ("9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411", 18657),
    "LICENSE-OFL-NotoSans.txt": ("cee9892f9f0cc8fe882c9e9537ee6a89621d86ee7ceaf70b02e2b2b1c25c061a", 4396),
    "LICENSE-OFL-NotoSansArabic.txt": ("a7a5a25eb188bf1cd96982030d53e23c33485c69b1044a562254226857ee13af", 4382),
    "NOTICE-translate.txt": ("13840d1654e6c954e62c5412447d1999a80c7e813c33fc3619b3fb1662d5800b", 1690),
    "NotoSans-Regular.ttf": ("478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823", 621572),
    "NotoSansArabic-Regular.ttf": ("bdff3e5659d67e67def05b33f749683b9376ae819d65d3dd62ac4640b3aaef48", 234892),
    "config.json": ("8f6496adfc930cbfecbe8281112197705c488fab47d34b4829b06d7f478909af", 223),
    "det.onnx": ("d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9", 4745517),
    "model.bin": ("990411b123076f5706de0b55a5ba1609322a18306c0e525043a86ab87634ae36", 78424155),
    "rec.onnx": ("ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea", 8967018),
    "shared_vocabulary.json": ("1f3c24e6f4e03183b20b95b757f3b7df30512bdf9980a6818c5393b4f8b9aaf6", 1815066),
    "source.spm": ("81ac8f4d4ee271f4109eac67560a31f9c2d555463ba5eb95c32c6deee8ae819d", 801074),
    "target.spm": ("67d0f6afb87768c078ee46c0756b8cc94b1ec23c5526b49ecf6c3b2f9197ec39", 916890),
}

# --- updates --------------------------------------------------------------------------------------------------
# A small JSON file signed with ed25519 (``tools/sign_manifest.py``): latest app version, model files and head
# files with sha256. Empty URL or no key: the app never looks for updates and works as it is. The private key
# never enters this repository.
UPDATE_MANIFEST_URL = ""
UPDATE_PUBLIC_KEYS: tuple[str, ...] = ()  # hex, 32 bytes each; more than one allows changing the key

# --- subscriptions (Dodo Payments licence keys, app/license.py) ------------------------------------------------
# The app talks only to Dodo's PUBLIC licence endpoints (activate / validate / deactivate: no API key, no secret in
# the app). LICENSE_REQUIRED False: the beta runs without a key (normal plan). Ali turns it on once the two
# products exist and their ids are filled in below.
LICENSE_REQUIRED = False
DODO_MODE = "test"  # "test" while trying it, "live" for real customers
DODO_BASE = {"test": "https://test.dodopayments.com", "live": "https://live.dodopayments.com"}
# Dodo product id -> plan. TEST-mode products (17 Nov 2026); live mode gets new ids after verification.
PRODUCT_PLANS: dict[str, str] = {
    "pdt_0Ns8bH6iyuBWR4ByTJk5V": "basic",  # Basic monthly 49 SAR
    "pdt_0Ns8udAcfdHoVMGkSgqJH": "basic",  # Basic yearly 470 SAR
    "pdt_0Ns8wMhWzDtbNeUffyLjs": "pro",  # Pro monthly 99 SAR
    "pdt_0Ns8wbYzBAuDWpHK6G7z8": "pro",  # Pro yearly 950 SAR
}
BUY_URL = "https://streamsafe-sa.vercel.app/#pricing"  # where a streamer subscribes
# launch period: Ali's support page; the Home tab shows a support card when set
SUPPORT_URL = "https://ko-fi.com/streamsafe"
# supporter passes (app/passkey.py): a key that turns Pro on for a fixed time. Made by tools/make_pass.py.
PASS_PUBLIC_KEYS: tuple[str, ...] = ("92cb956f0820177cd1ac1e268b503653c659878630db39baa4578e25aa6c0bdb",)
LICENSE_GRACE_DAYS = 3  # no internet: a key that was valid keeps working this long
LICENSE_RECHECK_HOURS = 6  # while protection runs, the key is checked again this often

# --- beta ---------------------------------------------------------------------------------------------------
# >0: a beta build stops starting streams this many days after it was built (``build_info.EXPIRES``).
# 0: never (the paid builds: the subscription decides).
BETA_DAYS = 0


# --- NVIDIA fast mode -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Wheel:
    """One vendor wheel (a zip) from the vendor's own package index, pinned by size and sha256."""

    name: str
    url: str
    sha256: str
    size: int
    keep: tuple[str, ...]  # path prefixes inside the wheel that are unpacked; the rest is not needed


_PYPI = "https://files.pythonhosted.org/packages"
# The pair run 57 measured (docs/NVIDIA_TRT.md): ONNX Runtime GPU 1.22.0 + TensorRT 10.9, with the CUDA 12.8 and
# cuDNN 9.8 runtimes ONNX Runtime names as its own extras (``onnxruntime-gpu[cuda,cudnn]``). Windows x64, CPython 3.11.
# Published by Microsoft (onnxruntime-gpu, MIT) and by NVIDIA (the rest, NVIDIA licences) on PyPI / pypi.nvidia.com.
NVIDIA_PYTHON = (3, 11)  # onnxruntime-gpu's extension module is built for exactly this Python
NVIDIA_WHEELS: tuple[Wheel, ...] = (
    Wheel(
        "onnxruntime-gpu 1.22.0",
        f"{_PYPI}/1e/47/313ee7998ef63dd7533200966972056fc5f3c7dd3bdfd9c49ae833bb5108/onnxruntime_gpu-1.22.0-cp311-cp311-win_amd64.whl",
        "0f1719f7cca76075b398a7d0466ead62d78fd2b8c0ea053dcf65d80c813103e8",
        214923507,
        ("onnxruntime/", "onnxruntime_gpu-1.22.0.dist-info/"),
    ),
    Wheel(
        "nvidia-cuda-runtime-cu12 12.8.90",
        f"{_PYPI}/30/a5/a515b7600ad361ea14bfa13fb4d6687abf500adc270f19e89849c0590492/nvidia_cuda_runtime_cu12-12.8.90-py3-none-win_amd64.whl",
        "c0c6027f01505bfed6c3b21ec546f69c687689aad5f1a377554bc6ca4aa993a8",
        944318,
        ("nvidia/", "nvidia_cuda_runtime_cu12-12.8.90.dist-info/"),
    ),
    Wheel(
        "nvidia-cublas-cu12 12.8.4.1",
        f"{_PYPI}/70/61/7d7b3c70186fb651d0fbd35b01dbfc8e755f69fd58f817f3d0f642df20c3/nvidia_cublas_cu12-12.8.4.1-py3-none-win_amd64.whl",
        "47e9b82132fa8d2b4944e708049229601448aaad7e6f296f630f2d1a32de35af",
        567544208,
        ("nvidia/", "nvidia_cublas_cu12-12.8.4.1.dist-info/"),
    ),
    Wheel(
        "nvidia-cufft-cu12 11.3.3.83",
        f"{_PYPI}/7d/ec/ce1629f1e478bb5ccd208986b5f9e0316a78538dd6ab1d0484f012f8e2a1/nvidia_cufft_cu12-11.3.3.83-py3-none-win_amd64.whl",
        "7a64a98ef2a7c47f905aaf8931b69a3a43f27c55530c698bb2ed7c75c0b42cb7",
        192216559,
        ("nvidia/", "nvidia_cufft_cu12-11.3.3.83.dist-info/"),
    ),
    Wheel(
        "nvidia-curand-cu12 10.3.9.90",
        f"{_PYPI}/b9/75/70c05b2f3ed5be3bb30b7102b6eb78e100da4bbf6944fd6725c012831cab/nvidia_curand_cu12-10.3.9.90-py3-none-win_amd64.whl",
        "f149a8ca457277da854f89cf282d6ef43176861926c7ac85b2a0fbd237c587ec",
        62765309,
        ("nvidia/", "nvidia_curand_cu12-10.3.9.90.dist-info/"),
    ),
    Wheel(
        "nvidia-cuda-nvrtc-cu12 12.8.93",
        f"{_PYPI}/45/51/52a3d84baa2136cc8df15500ad731d74d3a1114d4c123e043cb608d4a32b/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl",
        "7a4b6b2904850fe78e0bd179c4b655c404d4bb799ef03ddc60804247099ae909",
        73586838,
        ("nvidia/", "nvidia_cuda_nvrtc_cu12-12.8.93.dist-info/"),
    ),
    Wheel(
        "nvidia-cudnn-cu12 9.8.0.87",
        f"{_PYPI}/39/6a/5e9910b2b2c9dcddee9aaef372b3db0f08b9f7eaf1d462f859461a79caf9/nvidia_cudnn_cu12-9.8.0.87-py3-none-win_amd64.whl",
        "b4b5cfddc32aa4180f9d390ee99e9a9f55a89e7087329b41aba4319327e22466",
        684630375,
        ("nvidia/", "nvidia_cudnn_cu12-9.8.0.87.dist-info/"),
    ),
    Wheel(
        "tensorrt-cu12-libs 10.9.0.34",
        "https://pypi.nvidia.com/tensorrt-cu12-libs/tensorrt_cu12_libs-10.9.0.34-py2.py3-none-win_amd64.whl",
        "e43d38cc380d615bf8afab637c265f01a9452dc23deb9b9dd47b7662edf24531",
        0,  # NVIDIA's index does not publish the size; the sha256 decides
        ("tensorrt_libs/", "tensorrt_cu12_libs-10.9.0.34.dist-info/"),
    ),
)
NVIDIA_DOWNLOAD_GB = 3.5  # shown to the streamer before they say yes
NVIDIA_DISK_GB = 12.0  # free space asked for: the downloads, the unpacked libraries and the engines
NVIDIA_LICENSE_URLS = (
    "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html",
    "https://docs.nvidia.com/cuda/eula/index.html",
    "https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html",
)


# every host the app itself may contact (the stream output goes where the streamer points it)
def allowed_hosts() -> set[str]:
    from urllib.parse import urlsplit

    urls = [MODEL_BASE_URL, TRANSLATE_BASE_URL, UPDATE_MANIFEST_URL, *(w.url for w in NVIDIA_WHEELS)]
    return {urlsplit(u).hostname or "" for u in urls if u}
