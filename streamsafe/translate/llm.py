"""English -> Arabic by a small instruction model (GGUF, run by llama.cpp through ``llama-cpp-python``, MIT) on the GPU.

The strong translator. opus-mt (``translator.Ct2Translator``) translates sentence by sentence without knowing what a
game is; a 4-billion-parameter model that is told "this is a button prompt / a chat line / a subtitle of a video game"
keeps names and keys, understands slang and writes the Arabic a localiser would. It needs a graphics card with free
memory, so it is optional: ``choose`` (bottom) picks it only when it can run, and ``LlmGameTranslator`` hands every line
it is not sure about to the fallback translator or shows nothing.

Nothing here imports ``llama_cpp`` until a model is loaded; everything but ``LlamaTranslator`` is plain text code
(prompts, the check of the model's answer) and is tested without a model.
"""

from __future__ import annotations

import re
import subprocess
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from streamsafe.translate import gamefix, textproc

GGUF_FILE = "translate-llm.gguf"  # the name of the model file in the pack, whichever model it is
MIN_FREE_MB = 3500  # free video memory needed before the model is loaded in "auto" (the model takes about 3 GB)
N_CTX = 1024  # tokens: the fixed instructions (about 450) + one paragraph + its Arabic
MAX_NEW = 160

# ---- what kind of text is it ------------------------------------------------------------------------------------------
_HUD = re.compile(r"^(?:press|hold|tap|push|click|pull|release|hit|mash|double[- ]tap)\b", re.I)
_CHAT = re.compile(r"^(?:[\[<][^\]>]{1,14}[\]>]\s*)?[\w.-]{2,24}:\s")
_SOCIAL = re.compile(r"(?<!\w)[@#][A-Za-z_]\w+|https?://|www\.|\b[\w-]+\.(?:com|tv|gg|net|org)\b")
_SENTENCE = re.compile(r"[.?!,;…]")

# verbs that open an objective or a tip ("Secure the landing zone."), beside ``gamefix.IMPERATIVES``; several are nouns
# or adjectives too ("Save game", "Light armor"), so ``kind_of`` wants a full sentence after them
_ORDERS = frozenset(
    "activate aim brake build burn buy capture choose clear climb close combine cook craft deal decide defend defuse "
    "deliver disable drift eliminate enter equip escort exit free gather hack hide hunt investigate jump kill land leave "
    "light locate meet mine open parry place plant protect reach repair rescue restore rob save score search secure select "
    "sell speak steal survive swim talk track upgrade upload wait win brush".split()
)
KINDS = {
    "hud": "a button prompt",
    "objective": "a mission objective or tip, an order to the player",
    "social": "a chat line or social-media post",
    "ui": "a menu label, item name or notification",
    "subtitle": "a subtitle (spoken dialogue)",
}


def kind_of(text: str) -> str:
    """'hud' (button prompt), 'social' (chat, post, handle), 'objective' (an order to the player: "Lose the cops."),
    'ui' (label, item, notification) or 'subtitle'."""
    t = text.strip()
    if _HUD.match(t):
        return "hud"
    if _SOCIAL.search(t) or (_CHAT.match(t) and textproc.keep_tokens(t)):
        return "social"
    if "," not in t and len(textproc.sentences(t)) == 1:
        words = t.split()
        if gamefix._imperative_first(t) or (words[0].lower() in _ORDERS and (len(words) >= 4 or t[-1] in ".!")):
            return "objective"
    if len(t.split()) <= 5 and not _SENTENCE.search(t.rstrip(".!?…")):
        return "ui"
    return "subtitle"


# ---- prompts ----------------------------------------------------------------------------------------------------------
RULES = (
    "You translate the on-screen text of video games from English into Arabic for a live-stream overlay.\n"
    "- Write short, natural Modern Standard Arabic, the way a professional game localiser does.\n"
    "- Translate the meaning of slang, idioms and gamer talk; never translate them word by word.\n"
    "- Copy unchanged, in Latin letters: player names, @handles, #hashtags, web addresses, brand names, "
    "key and button names (E, TAB, SHIFT, [RT]) and numbers.\n"
    "- Orders to the player (objectives, tips, button prompts) start with a masculine singular imperative verb; "
    "\"Hold\" a key is \"اضغط مطولًا على\".\n"
    "- Answer with the Arabic translation only, on one line: no notes, no quotes, no English, no transliteration."
)
# (kind, English, Arabic): none of these lines is in a test set
SHOTS: tuple[tuple[str, str, str], ...] = (
    ("hud", "Press G to open the door", "اضغط G لفتح الباب"),
    ("hud", "Hold [RB] to charge the attack", "اضغط مطولًا على [RB] لشحن الهجوم"),
    ("hud", "Hold SPACE to climb", "اضغط مطولًا على SPACE للتسلق"),
    ("objective", "Find the key to the cellar.", "اعثر على مفتاح القبو."),
    ("objective", "Defend the gate until dawn.", "دافع عن البوابة حتى الفجر."),
    ("objective", "Drive the truck to the border.", "قُد الشاحنة إلى الحدود."),
    ("ui", "Quest completed", "اكتملت المهمة"),
    ("ui", "Steel dagger", "خنجر فولاذي"),
    ("ui", "Door unlocked", "فُتح الباب"),
    ("social", "@mike_1990 liked your photo", "أعجب @mike_1990 بصورتك"),
    ("social", "[TEAM] Rex_55: cover me, I'm reloading", "[TEAM] Rex_55: غطّوني، أعيد التلقيم"),
    ("subtitle", "Man, that was a close one.", "يا رجل، كدنا نهلك."),
    ("subtitle", "I'm not gonna ask you twice.", "لن أطلب منك مرتين."),
    ("subtitle", "Get in the truck, we're leaving.", "اركب الشاحنة، سنغادر."),
    ("subtitle", "This place is a dump, let's bounce.", "هذا المكان مزرٍ، لنرحل من هنا."),
)
TG_HEAD = (  # the instruction TranslateGemma was trained with (its chat template, en -> ar)
    "You are a professional English (en) to Arabic (ar) translator. Your goal is to accurately convey the meaning and "
    "nuances of the original English text while adhering to Arabic grammar, vocabulary, and cultural sensitivities.\n"
)
TG_TAIL = (
    "Produce only the Arabic translation, without any additional explanations or commentary. "
    "Please translate the following English text into Arabic:\n\n\n"
)
TG_GAME = (
    "The text is {kind} from the screen of a video game. Use short, natural Modern Standard Arabic; translate the meaning "
    "of slang, not the words. Copy player names, @handles, #hashtags, web addresses, brand names, key and button names "
    "(E, TAB, [RT]) and numbers unchanged in Latin letters. Orders to the player are masculine singular imperatives.\n"
)
STYLES = ("chatml", "gemma", "tgemma", "tgemma-game")
STOPS = {
    "chatml": ["<|im_end|>", "<|endoftext|>", "\n"],
    "gemma": ["<end_of_turn>", "\n"],
    "tgemma": ["<end_of_turn>", "\n"],
    "tgemma-game": ["<end_of_turn>", "\n"],
}


def _ask(kind: str, text: str) -> str:
    return f"[{KINDS[kind]}] {text}"


def build_prompt(text: str, style: str, kind: str | None = None) -> str:
    """The whole prompt for one line. Everything before the line itself is the same for every call of one style (and,
    for the chat styles, every kind): llama.cpp keeps that part's computation, a call pays only for the line."""
    kind = kind or kind_of(text)
    if style == "chatml":  # Qwen and other ChatML models; the empty think block switches a "thinking" model's reasoning off
        out = [f"<|im_start|>system\n{RULES}<|im_end|>\n"]
        for k, en, ar in SHOTS:
            out.append(f"<|im_start|>user\n{_ask(k, en)}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{ar}<|im_end|>\n")
        out.append(f"<|im_start|>user\n{_ask(kind, text)}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return "".join(out)
    if style == "gemma":  # Gemma has no system turn: the rules open the first user turn
        out = []
        for i, (k, en, ar) in enumerate(SHOTS):
            head = RULES + "\n\n" if i == 0 else ""
            out.append(f"<start_of_turn>user\n{head}{_ask(k, en)}<end_of_turn>\n<start_of_turn>model\n{ar}<end_of_turn>\n")
        out.append(f"<start_of_turn>user\n{_ask(kind, text)}<end_of_turn>\n<start_of_turn>model\n")
        return "".join(out)
    if style == "tgemma":
        return f"<start_of_turn>user\n{TG_HEAD}{TG_TAIL}{text}<end_of_turn>\n<start_of_turn>model\n"
    if style == "tgemma-game":
        return f"<start_of_turn>user\n{TG_HEAD}{TG_GAME.format(kind=KINDS[kind])}{TG_TAIL}{text}<end_of_turn>\n<start_of_turn>model\n"
    raise ValueError(f"unknown prompt style {style!r}")


def style_for(metadata: dict[str, Any]) -> str:
    """The prompt style of a GGUF file from what the file says about itself."""
    name = " ".join(str(metadata.get(k, "")) for k in ("general.name", "general.basename", "general.architecture")).lower()
    template = str(metadata.get("tokenizer.chat_template", ""))
    if "translategemma" in name.replace(" ", "").replace("-", "").replace("_", ""):
        return "tgemma-game"
    if "<start_of_turn>" in template or "gemma" in name:
        return "gemma"
    return "chatml"


# ---- the answer -------------------------------------------------------------------------------------------------------
_TAG = re.compile(r"^\s*\[(?:" + "|".join(map(re.escape, KINDS.values())) + r")\]\s*")  # the model repeated the "[a subtitle ...]" label
_WRAP = re.compile(r"^[\"'\u00ab\u201c\u2018`*]+|[\"'\u00bb\u201d\u2019`*]+$")
_LATIN = re.compile(r"[A-Za-z][A-Za-z'’]*")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_HARAKAT = re.compile("[\u064b-\u0650\u0652\u0670]")  # not the shadda: "\u063a\u0637\u0651\u0648\u0646\u064a"
_ARABIC = re.compile(r"[\u0621-\u064a]")
_LOOP = re.compile(r"(\b[\w\u0600-\u06ff]+\b)(?:\s+\1){3,}")  # one word again and again


def clean_answer(en: str, raw: str) -> str:
    """The Arabic line out of the model's raw answer, or "" when the answer is not a translation of ``en`` we can show:
    no Arabic, a second script (a model that falls back into Chinese or Russian), English that is not in the source
    (a note, the sentence left untranslated), a name / handle / address / button of the source missing or respelled,
    a loop, or far too long. "" is always safe: the caller uses the fallback translator or shows nothing."""
    ar = raw.replace("\r", "\n").strip().split("\n")[0].strip()
    ar = _TAG.sub("", ar)
    if '"' not in en and "'" not in en:
        ar = _WRAP.sub("", ar).strip()
    ar = _HARAKAT.sub("", ar.translate(_DIGITS)).replace("\u0640", "")
    ar = re.sub(r"\s+", " ", ar).strip()
    if not ar or not _ARABIC.search(ar):
        return ""
    for c in ar:
        if c.isalpha() and not c.isascii() and not ("\u0600" <= c <= "\u06ff" or "\ufb50" <= c <= "\ufefc"):
            return ""
        if unicodedata.category(c) in ("Co", "Cn", "So") and c not in "×":
            return ""
    source = {w.lower() for w in _LATIN.findall(en)}
    if any(w.lower() not in source for w in _LATIN.findall(ar) if len(w) > 1):
        return ""
    if any(tok not in ar for tok in textproc.keep_tokens(en)):
        return ""
    if _LOOP.search(ar) or len(ar) > 3 * len(en) + 24:
        return ""
    return ar


# ---- the model --------------------------------------------------------------------------------------------------------
class LlamaTranslator:
    """One GGUF model in this process. ``translate`` returns "" for a line whose answer does not pass ``clean_answer``."""

    def __init__(self, gguf: Path, style: str | None = None, gpu_layers: int = -1, threads: int = 2, n_ctx: int = N_CTX) -> None:
        from llama_cpp import Llama

        self.model = Llama(
            model_path=str(gguf),
            n_gpu_layers=gpu_layers,
            n_ctx=n_ctx,
            n_batch=512,
            n_threads=threads,
            n_threads_batch=threads,
            seed=0,
            verbose=False,
        )
        self.style = style or style_for(dict(self.model.metadata or {}))
        self.raw: list[tuple[str, str]] = []  # the last answers as the model gave them (bench, debugging); bounded

    def answer(self, text: str, kind: str | None = None) -> str:
        prompt = build_prompt(text, self.style, kind)
        room = self.model.n_ctx() - len(self.model.tokenize(prompt.encode("utf-8"), special=True)) - 8
        if room < 16:
            return ""  # a paragraph longer than the context: not ours to translate
        out = self.model.create_completion(
            prompt,
            max_tokens=min(MAX_NEW, room, 24 + 3 * len(text)),
            temperature=0.0,
            top_k=1,
            repeat_penalty=1.0,
            stop=STOPS[self.style],
        )
        return out["choices"][0]["text"]

    def translate(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            raw = self.answer(t) if t.strip() else ""
            self.raw.append((t, raw))
            del self.raw[:-64]
            out.append(clean_answer(t, raw))
        return out


class LlmGameTranslator:
    """The strong model with the game rules that need no model, and a net under it.

    A line of the fixed glossary (``gamefix.FIXED``: "WASTED", "Mission passed") gets its fixed Arabic. Every other line
    goes to ``llm`` whole (a paragraph is not cut into sentences: the model uses the context). A line the model fails on
    (``clean_answer``, or an exception) goes to ``fallback`` (opus-mt with its game rules) or, without one, gets "":
    no overlay for that line. After ``max_failures`` exceptions in a row the model is given up for good."""

    def __init__(self, llm: Any, fallback: Any = None, max_failures: int = 3, on_give_up: Callable[[str], None] | None = None) -> None:
        self.llm, self.fallback, self.max_failures, self.on_give_up = llm, fallback, max_failures, on_give_up
        self.failures = 0
        self.stats = {"fixed": 0, "llm": 0, "fallback": 0, "empty": 0}

    def _fixed(self, text: str) -> str:
        from streamsafe.translate import gamefix  # here, not at the top: the prompts above are used without it (bench)

        fixed = gamefix.FIXED.get(textproc.cache_key(text))
        return gamefix.finish(gamefix.Job(textproc.clean(text), fixed=fixed), "") or "" if fixed else ""

    def translate(self, texts: list[str]) -> list[str]:
        out = [self._fixed(t) for t in texts]
        self.stats["fixed"] += sum(1 for o in out if o)
        todo = [i for i, o in enumerate(out) if not o and texts[i].strip()]
        if todo and self.llm is not None:
            try:
                done = self.llm.translate([textproc.clean(texts[i]) for i in todo])
                self.failures = 0
            except Exception as exc:  # the model library failed: these lines go to the fallback
                done = [""] * len(todo)
                self.failures += 1
                if self.failures >= self.max_failures:
                    self.llm = None
                    if self.on_give_up:
                        self.on_give_up(f"{exc!r:.200}")
            for i, ar in zip(todo, done, strict=True):
                out[i] = ar
                self.stats["llm"] += bool(ar)
        rest = [i for i in todo if not out[i]]
        if rest and self.fallback is not None:
            try:
                for i, ar in zip(rest, self.fallback.translate([texts[i] for i in rest]), strict=True):
                    out[i] = ar
                    self.stats["fallback"] += bool(ar)
            except Exception:
                pass
        self.stats["empty"] += sum(1 for i in todo if not out[i])
        return out


# ---- may it run here --------------------------------------------------------------------------------------------------
def free_vram_mb(run: Callable[..., Any] = subprocess.run) -> int | None:
    """Free memory of the NVIDIA card with the most of it, in MB; None when it cannot be read (no NVIDIA card or
    driver tool). Never raises."""
    try:
        r = run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
        return max(vals) if r.returncode == 0 and vals else None
    except Exception:
        return None


def gpu_offload() -> bool:
    """Is ``llama-cpp-python`` installed AND built with a GPU backend (CUDA, Vulkan, ...)?"""
    try:
        import llama_cpp

        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def choose(
    mode: str,
    gguf: Path,
    min_free_mb: int = MIN_FREE_MB,
    free: Callable[[], int | None] = free_vram_mb,
    offload: Callable[[], bool] = gpu_offload,
) -> tuple[bool, str]:
    """(use the strong model?, why) for ``translate_model`` = 'auto' | 'llm' | 'opus'.

    auto: only with the model file, a GPU build of llama.cpp and enough free video memory right now (the game and the
    protection models come first; on the CPU this model is far too slow for a live stream). llm: whenever the file and
    the library are there (the streamer's own decision). The load itself can still fail: the caller falls back."""
    mode = (mode or "auto").strip().lower()
    if mode == "opus":
        return False, "translate_model=opus"
    if mode not in ("auto", "llm"):
        return False, f"unknown translate_model {mode!r}"
    if not gguf.is_file():
        return False, "the strong translation model is not installed"
    if mode == "llm":
        try:
            import llama_cpp  # noqa: F401
        except Exception as exc:
            return False, f"llama-cpp-python is not usable ({exc!r:.120})"
        return True, "translate_model=llm"
    if not offload():
        return False, "llama-cpp-python with GPU support is not installed"
    mb = free()
    if mb is None:
        return False, "free video memory could not be read"
    if mb < min_free_mb:
        return False, f"only {mb} MB of video memory free (needs {min_free_mb})"
    return True, f"{mb} MB of video memory free"
