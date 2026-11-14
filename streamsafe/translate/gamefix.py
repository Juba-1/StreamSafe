"""Game-text rules around the translation model: what to protect before it, what to repair after it. No model, no I/O.

The model (opus-mt) is good at sentences and bad at what game screens are made of: key names ("Press E" came back
as "اضغط على (إي)", ENTER as "المُدخل"), amounts ("$15,000" as "000 15"), ALL-CAPS and Title Case titles (garbage), place
names ("Vice City" as "نائب المدينة"), fixed UI words, and imperatives in the feminine ("اركبي"). Everything here is
deterministic and costs microseconds; the model is called once per sentence, as before.

``plan`` turns one English sentence into a ``Job``: either a fixed answer (glossary), a key prompt (head + the model's
translation of the action), or a sentence for the model with some spans swapped for stand-ins that the model keeps
(numbers, whole place names). ``finish`` restores the stand-ins and cleans the Arabic. A lost stand-in makes ``finish``
return ``None``: the caller asks the model again for the plain sentence (``plan(..., protect=False)``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from streamsafe.translate import textproc

# ---- glossary ---------------------------------------------------------------------------------------------------------

# whole lines (matched on ``textproc.cache_key``): fixed UI words and the stock lines of the genre
FIXED: dict[str, str] = {
    "resume": "استئناف",
    "settings": "الإعدادات",
    "map": "الخريطة",
    "inventory": "الحقيبة",
    "weapons": "الأسلحة",
    "save game": "حفظ اللعبة",
    "load game": "تحميل اللعبة",
    "new game": "لعبة جديدة",
    "quit to menu": "الخروج إلى القائمة",
    "quit game": "الخروج من اللعبة",
    "back": "رجوع",
    "options": "الخيارات",
    "phone": "الهاتف",
    "stats": "الإحصائيات",
    "brief": "الملخص",
    "controls": "عناصر التحكم",
    "audio": "الصوت",
    "display": "العرض",
    "graphics": "الرسوميات",
    "pause": "إيقاف مؤقت",
    "start": "ابدأ",
    "loading": "جارٍ التحميل...",
    "continue": "متابعة",
    "confirm": "تأكيد",
    "cancel": "إلغاء",
    "accept": "قبول",
    "decline": "رفض",
    "yes": "نعم",
    "no": "لا",
    "ok": "حسنًا",
    "wasted": "قُتلت",
    "busted": "تم القبض عليك",
    "mission failed": "فشلت المهمة",
    "mission passed": "نجحت المهمة",
    "mission complete": "اكتملت المهمة",
    "mission completed": "اكتملت المهمة",
    "game saved": "تم حفظ اللعبة",
    "game over": "انتهت اللعبة",
    "checkpoint reached": "تم الوصول إلى نقطة التحقق",
    "objective updated": "تم تحديث الهدف",
    "new objective": "هدف جديد",
    "wanted level": "مستوى الملاحقة",
    "wanted level cleared": "تمت إزالة مستوى الملاحقة",
    "waypoint set": "تم تحديد نقطة الوجهة",
    "new waypoint set": "تم تحديد نقطة وجهة جديدة",
    "you have been busted": "تم القبض عليك",
    "you are busted": "تم القبض عليك",
    "you have been arrested": "تم اعتقالك",
    "you lost the cops": "لقد هربت من الشرطة",
    "you are wanted by the police": "أنت مطلوب من الشرطة",
    "you are being chased by the police": "الشرطة تطاردك",
    "not enough money": "لا يوجد مال كافٍ",
    "press any key": "اضغط أي مفتاح",
    "let's roll": "هيا بنا",
    "lets roll": "هيا بنا",
    "rockstar games presents": "روكستار جيمز تقدّم",
    "lose them": "اهرب منهم",
    "lose him": "اهرب منه",
    "lose her": "اهرب منها",
    "you good": "هل أنت بخير؟",
    "we did it": "لقد نجحنا!",
}

# names that must come out as written here: the whole name is swapped for a city the model always spells the same way
_STANDINS = {
    "Boston": "بوسطن",
    "Chicago": "شيكاغو",
    "Houston": "هيوستن",
    "Dallas": "دالاس",
    "Seattle": "سياتل",
    "Phoenix": "فينيكس",
}
NAMES: dict[str, str] = {
    "rockstar games": "روكستار جيمز",
    "rockstar": "روكستار",
    "grand theft auto": "غراند ثيفت أوتو",
    "vice city": "فايس سيتي",
    "vice beach": "شاطئ فايس",
    "leonida keys": "ليونيدا كيز",
    "leonida": "ليونيدا",
    "port gellhorn": "بورت جيلهورن",
    "grassrivers": "غراسريفرز",
    "ambrosia": "أمبروزيا",
    "kalaga national park": "منتزه كالاغا الوطني",
    "kalaga": "كالاغا",
}
_NAME_RE = re.compile(r"(?:\b(?:the)\s+)?\b(" + "|".join(sorted(map(re.escape, NAMES), key=len, reverse=True)) + r")\b", re.I)
# the model's spellings of people the game is about -> the spelling used on screen
SPELLING: dict[str, str] = {"جايسون": "جيسون", "جيسن": "جيسون", "لوسيا": "لوسيا", "لوشيا": "لوسيا"}

# terms: when the English has the term, these renderings of the model are replaced by the fixed one
TERMS: list[tuple[re.Pattern[str], list[tuple[re.Pattern[str], str]]]] = [
    (
        re.compile(r"\bwanted\s+levels?\b", re.I),
        [(re.compile(r"(?:ال)?مستوى\s+(?:ال)?مطلوب(?:ين|ك|ة)?"), "مستوى الملاحقة")],
    ),
    (re.compile(r"\bmissions?\b", re.I), [(re.compile(r"\bالبعثة\b"), "المهمة"), (re.compile(r"\bبعثة\b"), "مهمة")]),
    (re.compile(r"\bwelcome\b", re.I), [(re.compile(r"\bمرحبا بكم\b"), "مرحبًا بك"), (re.compile(r"\bمرحبا\b"), "مرحبًا")]),
    (
        re.compile(r"\bbank\b", re.I),
        [(re.compile(r"\bالرصيد المصرفي\b"), "رصيد البنك"), (re.compile(r"\bالمصرف(?:ي)?\b"), "البنك"), (re.compile(r"\bمصرف\b"), "بنك")],
    ),
    (re.compile(r"\bmarina\b", re.I), [(re.compile(r"\bالميناء\b"), "المرسى"), (re.compile(r"\bمارينا\b"), "المرسى")]),
    (re.compile(r"\bwaypoints?\b", re.I), [(re.compile(r"نقطة\s+(?:ال)?طريق"), "نقطة الوجهة")]),
    (re.compile(r"\bsafe ?houses?\b", re.I), [(re.compile(r"(?:ال)?(?:منزل|بيت) (?:ال)?آمن"), "المخبأ")]),
    (re.compile(r"\bgarages?\b", re.I), [(re.compile(r"\bالمرآب\b"), "الكراج")]),
    (re.compile(r"\bcops?\b", re.I), [(re.compile(r"\bرجال الشرطة\b"), "الشرطة")]),
]

# actions of key prompts the model gets wrong as a bare verb phrase (matched on cache_key of the action)
ACTIONS: dict[str, str] = {
    "take cover": "للاحتماء",
    "crouch": "للانحناء",
    "honk the horn": "لإطلاق البوق",
    "honk": "لإطلاق البوق",
    "refuel": "لتعبئة الوقود",
    "reload": "لإعادة التعبئة",
    "aim": "للتصويب",
    "fire": "لإطلاق النار",
    "continue": "للمتابعة",
    "exit the vehicle": "للنزول من السيارة",
    "enter the vehicle": "لركوب السيارة",
    "get in the vehicle": "لركوب السيارة",
    "get in the car": "لركوب السيارة",
    "exit the car": "للنزول من السيارة",
}

# slang the model has never seen -> plain English it translates well (whole words, case kept)
SLANG: dict[str, str] = {
    "gotta": "have to",
    "gonna": "going to",
    "wanna": "want to",
    "gimme": "give me",
    "lemme": "let me",
    "y'all": "you all",
    "cuz": "because",
    "homie": "friend",
    "homies": "friends",
    "yo": "hey",
    "babe": "darling",
    "nah": "no",
    "bro": "brother",
    "dude": "man",
    "ya": "you",
}
_SLANG = re.compile(r"\b(" + "|".join(map(re.escape, SLANG)) + r")\b", re.I)


def _slang(text: str) -> str:
    def fix(m: re.Match[str]) -> str:
        w = SLANG[m.group(1).lower()]
        return w[:1].upper() + w[1:] if m.group(1)[:1].isupper() else w

    return _SLANG.sub(fix, text)


# ---- case ---------------------------------------------------------------------------------------------------------------

SMALL = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "nor",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "with",
        "by",
        "as",
        "into",
        "onto",
        "off",
        "up",
        "out",
        "over",
        "via",
        "per",
        "vs",
    ]
)
ACRONYMS = frozenset(
    [
        "ATM",
        "FBI",
        "LSPD",
        "GPS",
        "HQ",
        "UI",
        "VIP",
        "TV",
        "DJ",
        "NPC",
        "SUV",
        "ID",
        "ATV",
        "US",
        "USA",
        "UK",
        "EMS",
        "SWAT",
        "DEA",
        "CEO",
        "PIN",
        "OK",
        "HUD",
        "AI",
        "GTA",
        "PC",
    ]
)
KNOWN_CAPS = frozenset({"Lucia", "Jason", "Vice", "Leonida", "Rockstar"}) | frozenset(_STANDINS)
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")
_I = re.compile(r"\bi(?=\b|'(?:m|ll|ve|d)\b)")


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def shouting(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) >= 0.9 * len(letters)


def titled(text: str) -> bool:
    """Title Case: every word starts with a capital (or is a small word) and capitalised small words show it is not
    just a sentence with names in it ("Go To The Airport", "Mission Failed")."""
    ws = _words(text)
    if len(ws) < 2 or shouting(text):
        return False
    if not all(w[0].isupper() or w.lower() in SMALL for w in ws):
        return False
    return sum(1 for w in ws[1:] if w.lower() in SMALL and w[0].isupper()) >= 1 or all(w[0].isupper() for w in ws)


def sentence_case(text: str, common: Callable[[str], bool] | None = None) -> str:
    """ALL CAPS and Title Case -> sentence case (the model reads sentence case far better). Words that look like names
    (not an everyday word of the model's vocabulary) and the acronyms of the game keep their capitals."""
    if not (shouting(text) or titled(text)):
        return text
    first = [True]

    def fix(m: re.Match[str]) -> str:
        w = m.group(0)
        low = w.lower()
        if w.upper() in ACRONYMS and len(w) > 1 and w.isupper():
            out = w.upper()
        elif w in KNOWN_CAPS or w.title() in KNOWN_CAPS or (common is not None and not common(low) and low not in SMALL):
            out = w.title()
        else:
            out = low
        if first[0]:
            out = out[:1].upper() + out[1:]
            first[0] = False
        return out

    out = _WORD.sub(fix, text)
    return _I.sub("I", out)


# ---- numbers --------------------------------------------------------------------------------------------------------------

_NUM = r"\d+(?:\.\d+)?"
_UNITS = {"km/h": "كم/س", "kph": "كم/س", "mph": "ميل/س", "km": "كم", "m": "م", "kg": "كغ", "ft": "قدم", "mi": "ميل"}
_SPAN = re.compile(
    r"(?P<time>(?<![\d:.,])\d{1,2}:\d{2}(?::\d{2})?(?![\d:]))"
    r"|(?P<pct>(?P<pctn>" + _NUM + r")\s?%)"
    r"|(?P<unit>(?P<unitn>" + _NUM + r")\s?(?P<u>km/h|kph|mph|km|kg|ft|mi|m)\b(?!\w))"
    r"|(?P<money>\$\s?(?P<moneyn>\d(?:[\d,]*\d)?(?:\.\d+)?)(?:\s?(?P<scale>million|billion|thousand|[kmb])\b(?!\w))?)"
    r"|(?P<big>(?<![\d.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d|,\d)|(?<![\d.,])\d{4,}(?:\.\d+)?(?!\d|,\d))",
    re.I,
)
_SCALE = {"k": "thousand", "m": "million", "b": "billion", "thousand": "thousand", "million": "million", "billion": "billion"}
_NUMBER_STANDINS = ("77", "88", "66", "55", "44", "33", "22", "99")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٪", "0123456789%")


# ---- key prompts ------------------------------------------------------------------------------------------------------

_KEYNAME = r"(?:enter|shift|space|spacebar|tab|esc|escape|ctrl|alt|backspace|delete|caps lock)"
_KEY = (
    r"(?:~[A-Z][A-Z_0-9]*~|\[[A-Za-z0-9 ]{1,10}\]|<[A-Za-z0-9 ]{1,10}>|F(?:[1-9]|1[0-2])\b|[A-Z0-9]\b"
    r"|(?:LMB|RMB|MMB|LT|RT|LB|RB|[LR][123])\b"
    r"|(?i:" + _KEYNAME + r")\b|(?i:(?:the )?(?:left|right|middle) mouse button)|(?i:any key))"
)
_PROMPT = re.compile(
    r"^(?P<verb>press and hold|hold down|hold|press|tap|double[- ]tap|mash|release)\s+"
    r"(?P<keys>" + _KEY + r"(?:\s*(?:,|/|\+|\bor\b|\band\b)\s*" + _KEY + r")*)\s+(?P<prep>to)\s+(?P<act>.+?)\s*[.!]*$",
    re.I,
)
_BARE = re.compile(
    r"^(?P<verb>press and hold|hold down|hold|press|tap|double[- ]tap|mash|release)\s+(?P<keys>"
    + _KEY
    + r"(?:\s*(?:,|/|\+|\bor\b|\band\b)\s*"
    + _KEY
    + r")*)\s*[.!]*$",
    re.I,
)
_KEY_SPLIT = re.compile(r"\s*(,|/|\+|\bor\b|\band\b)\s*", re.I)
_VERBS = {
    "press": "اضغط",
    "press and hold": "اضغط مع الاستمرار على",
    "hold": "اضغط مع الاستمرار على",
    "hold down": "اضغط مع الاستمرار على",
    "tap": "انقر",
    "double tap": "انقر مرتين",
    "mash": "اضغط بسرعة على",
    "release": "حرّر",
}
_MOUSE = {"left": "زر الماوس الأيسر", "right": "زر الماوس الأيمن", "middle": "زر الماوس الأوسط"}
_JOIN = {",": " و", "and": " و", "or": " أو", "/": " أو", "+": "+"}


def _key_text(k: str) -> str:
    low = k.lower().strip()
    if low == "any key":
        return "أي مفتاح"
    m = re.search(r"(left|right|middle) mouse button", low)
    if m:
        return _MOUSE[m.group(1)]
    if low in ("escape", "esc"):
        return "ESC"
    if low == "spacebar":
        return "SPACE"
    if k[0] in "[<":
        k = k[1:-1].strip()
    return k.upper() if re.fullmatch(r"[A-Za-z0-9 ]+", k) else k


def _keys(keys: str) -> str:
    out = ""
    parts = _KEY_SPLIT.split(keys.strip())
    for i, p in enumerate(parts):
        if i % 2 == 0:
            out += _key_text(p)
        else:
            sep = p.lower()
            joiner = _JOIN[sep]
            out += joiner if joiner == "+" else joiner + " "
    return out


# ---- the job ----------------------------------------------------------------------------------------------------------------

IMPERATIVES = frozenset(
    [
        "go",
        "get",
        "follow",
        "meet",
        "drive",
        "wait",
        "escape",
        "kill",
        "eliminate",
        "collect",
        "return",
        "pick",
        "deliver",
        "hide",
        "take",
        "hack",
        "break",
        "destroy",
        "defeat",
        "steal",
        "rob",
        "lose",
        "park",
        "talk",
        "sneak",
        "shoot",
        "climb",
        "find",
        "stay",
        "call",
        "ask",
        "press",
        "hold",
        "use",
        "stop",
        "give",
        "keep",
        "come",
        "run",
        "fly",
        "jump",
        "swim",
        "sell",
        "buy",
        "open",
        "close",
        "enter",
        "exit",
        "leave",
        "reach",
        "protect",
        "rescue",
        "chase",
        "shake",
        "avoid",
        "clear",
        "complete",
        "start",
        "finish",
        "wait",
        "grab",
        "bring",
        "let",
        "don't",
        "do",
        "search",
        "look",
        "watch",
        "listen",
        "check",
        "destroy",
        "blow",
        "set",
        "plant",
        "cut",
        "cross",
        "turn",
    ]
)
_FEM = {"عودي": "عد", "قودي": "قد", "قومي": "قم", "خذي": "خذ", "ابقي": "ابق", "كوني": "كن"}
_KEEP_I = frozenset({"إلى", "إذا", "إن", "إنه", "إنها", "إنهم", "إلا"})


@dataclass
class Job:
    text: str  # the English sentence, cleaned
    fixed: str = ""  # answer without the model
    src: str = ""  # what the model translates ("" = nothing to translate)
    head: str = ""  # key prompt: Arabic before the action's translation
    action: bool = False  # src is "In order to ..." and only its tail is used
    slots: list[tuple[str, str]] = field(default_factory=list)  # (stand-in, final text)
    imperative: bool = False
    please: bool = False  # src has a "please" the model turns into one more word to drop (it makes it answer in the imperative)
    np: bool = False  # src is a noun phrase: head + its translation


def _protect(text: str) -> tuple[str, list[tuple[str, str]]]:
    slots: list[tuple[str, str]] = []
    nums = [n for n in _NUMBER_STANDINS if n not in text]
    cities = [c for c in _STANDINS if c.lower() not in text.lower()]

    def name(m: re.Match[str]) -> str:
        if not cities:
            return m.group(0)
        city = cities.pop(0)
        slots.append((_STANDINS[city], NAMES[m.group(1).lower()]))
        return city

    text = _NAME_RE.sub(name, text)

    def span(m: re.Match[str]) -> str:
        if not nums:
            return m.group(0)
        stand = nums.pop(0)
        if m.group("time"):
            slots.append((stand, m.group("time")))
            return stand
        if m.group("pct"):
            slots.append((stand, m.group("pctn") + "%"))
            return stand
        if m.group("unit"):
            slots.append((stand, f"{m.group('unitn')} {_UNITS[m.group('u').lower()]}"))
            return stand
        if m.group("money"):
            scale = _SCALE.get((m.group("scale") or "").lower(), "")
            slots.append((stand, m.group("moneyn")))
            return f"{stand} {scale + ' ' if scale else ''}dollars"
        slots.append((stand, m.group("big")))
        return stand

    return _SPAN.sub(span, text), slots


# "Lose the cops": a noun phrase after a fixed verb the model gets wrong
TEMPLATES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^lose\s+(?:the\s+)?(?!(?:them|him|her|it|us|me)\b)(?P<np>.+?)\s*[.!]*$", re.I), "اهرب من")
]
_VOCATIVE = re.compile(r"^(?P<voc>[A-Z][\w']*(?:\s[A-Z][\w']*)?,\s+)(?P<rest>.+)$", re.S)
_AUX = frozenset(
    [
        "is",
        "are",
        "was",
        "were",
        "will",
        "would",
        "can",
        "could",
        "has",
        "have",
        "had",
        "does",
        "did",
        "must",
        "should",
        "may",
        "might",
        "am",
    ]
)


def _imperative_first(text: str) -> bool:
    """The sentence starts with a verb that gives an order (not "Park is closed", not "Start menu")."""
    ws = _words(text)
    return bool(ws) and ws[0].lower() in IMPERATIVES and (len(ws) == 1 or ws[1].lower() not in _AUX)


def plan(sentence: str, common: Callable[[str], bool] | None = None, glossary: bool = True, protect: bool = True) -> Job:
    """The job for one English sentence. ``glossary=False``: only the stand-ins and the case repair (for measuring);
    ``protect=False``: no stand-ins (the retry after one was lost)."""
    text = re.sub(r"~[a-z]~", "", textproc.clean(sentence)).strip()
    job = Job(text)
    if not text:
        return job
    prot = _protect if protect else (lambda t: (t, []))
    if glossary:
        fixed = FIXED.get(textproc.cache_key(text))
        if fixed:
            job.fixed = fixed
            return job
        b = _BARE.match(text)
        if b:
            job.fixed = _VERBS[re.sub(r"[- ]+", " ", b.group("verb").lower())] + " " + _keys(b.group("keys"))
            return job
        m = _PROMPT.match(text)
        if m:
            verb = re.sub(r"[- ]+", " ", m.group("verb").lower())
            act = sentence_case(m.group("act"), common)
            job.head = _VERBS[verb] + " " + _keys(m.group("keys"))
            known = ACTIONS.get(textproc.cache_key(act))
            if known:
                job.fixed = job.head + " " + known
                return job
            src, job.slots = prot(act)
            job.src, job.action = "In order to " + src[:1].lower() + src[1:] + ".", True
            return job
    text = sentence_case(text, common)
    if glossary:
        text = _slang(text)
        for pat, head in TEMPLATES:
            t = pat.match(text)
            if t:
                np, job.slots = prot(t.group("np"))
                job.head, job.np, job.src, job.imperative = head, True, "The " + np + ".", True
                return job
        job.imperative = _imperative_first(text)
        if job.imperative:
            job.please = True
            text = LEAD + " " + text[:1].lower() + text[1:]
        else:
            v = _VOCATIVE.match(text)
            if v and _imperative_first(v.group("rest")):
                job.imperative = job.please = True
                rest = v.group("rest")
                text = v.group("voc") + LEAD.lower() + " " + rest
    job.src, job.slots = prot(text)
    return job


_ORDER = ("من أجل", "لكي")
_FEW = frozenset(map(str, range(3, 11)))  # "5 دولارات" is right: three to ten take the plural
_PAIR = re.compile(r"\(([^()]*)\)")
_QUOTES = re.compile(r'["«»“”]')
_MARKS = re.compile(
    "[\u064b-\u0650\u0652-\u065f\u0670\u0640\u2047\ufffd]"
)  # harakat (not the shadda), tatweel, the model's "unknown" glyph
_REPEAT = re.compile(r"(\b[\w\u0600-\u06ff]+\b)(?:\s+\1){3,}")
_PLEASE = re.compile(r"(?:^|(?<=[،,]\s))(?:و?رجاء[ًا]?|الرجاء|يرجى|من فضلك|فضلا|فضلاً|لطفا|لطفاً|أرجوك|ارجوك|حسنا|حسناً|حسنًا|حسن|هيا)[،,]?\s*")
LEAD = "Please"  # the word put in front of an imperative: the model then answers with a verb ("اسرق"), not a noun ("سرقة")
_HAMZA = [
    (re.compile(r"\bالإ(ختباء|تصال|ستخدام|نتظار|ستمرار|قتحام|ستعداد)"), r"الا\1"),
    (re.compile(r"\b(ب|ل|و|ك)الإ(ختباء|تصال|ستخدام|نتظار|ستمرار|قتحام)"), r"\1الا\2"),
    (re.compile(r"\bاختبيء\b"), "اختبئ"),
    (re.compile(r"\bأتبع\b"), "اتبع"),
]
_FORM1 = frozenset(
    [
        "أهرب",
        "أسرق",
        "أقتل",
        "أخرج",
        "أدخل",
        "أذهب",
        "أترك",
        "أبحث",
        "أجمع",
        "أنتظر",
        "أحمل",
        "أركب",
        "أقفز",
        "أنزل",
        "أفتح",
        "أقف",
        "أجلس",
        "أركض",
        "أسبح",
        "أقود",
        "أخذ",
        "أبق",
        "أدفع",
    ]
)
_FEM = {"عودي": "عد", "قودي": "قد", "قومي": "قم", "خذي": "خذ", "ابقي": "ابق", "كوني": "كن", "قد": "قُد"}
_KEEP_I = frozenset({"إلى", "إذا", "إن", "إنه", "إنها", "إنهم", "إلا"})
_END = re.compile(r"[.!?؟…،,:]+$")


def _imperative(ar: str) -> str:
    """The model addresses a woman ("اركبي") or a crowd ("احتموا"); the game addresses the player: masculine singular."""
    words = ar.split(" ")
    for i, w in enumerate(words):
        bare = w[1:] if i and w.startswith("و") and len(w) > 4 else w
        pre = w[: len(w) - len(bare)]
        if i and not pre:
            continue
        if bare in _FEM:
            bare = _FEM[bare]
        elif len(bare) >= 4 and bare[0] in "اإأتعقخذسبدفنهكيحجرزطمص" and not bare.endswith("ني"):
            if bare.endswith("وا"):
                bare = bare[:-2]
            elif bare.endswith("ي") and (bare[0] in "اإأ" or (bare[0] == "ت" and len(bare) >= 5)):
                bare = bare[:-1]
        if i == 0 and bare[:1] == "إ" and bare not in _KEEP_I and len(bare) > 3:
            bare = "ا" + bare[1:]
        if i == 0 and bare in _FORM1:
            bare = "ا" + bare[1:]
        words[i] = pre + bare
    return " ".join(words)


def _restore(ar: str, slots: list[tuple[str, str]]) -> str | None:
    ar = ar.translate(_DIGITS)
    for stand, final in slots:
        if stand.isdigit():
            pat = re.compile(rf"(?<![\d.,]){stand}(?!\d|,\d|\.\d)" + (r"(?:\s?%)?" if final.endswith("%") else ""))
            if not pat.search(ar):
                return None
            ar = pat.sub(lambda _m, f=final: f, ar, count=1)
        else:
            if stand not in ar:
                return None
            ar = ar.replace(stand, final, 1)
    return ar


def _punct(text: str) -> str:
    """The sentence's closing punctuation, in Arabic form (the model drops or changes it)."""
    m = re.search(r"[.!?…]+$", text.strip())
    return m.group(0).replace("?", "؟") if m else ""


def _clean(ar: str, job: Job) -> str:
    ar = _MARKS.sub("", ar)
    if "(" not in job.text:  # the model puts every name in brackets
        ar = _PAIR.sub(r"\1", ar).replace("(", "").replace(")", "")
    if '"' not in job.text:
        ar = _QUOTES.sub("", ar)
    if job.please:
        ar = _PLEASE.sub("", ar, count=1)
    for pat, rep in _HAMZA:
        ar = pat.sub(rep, ar)
    if job.imperative:
        ar = _imperative(ar)
    for src_pat, fixes in TERMS:
        if src_pat.search(job.text):
            for pat, rep in fixes:
                ar = pat.sub(rep, ar)
    for bad, good in SPELLING.items():
        ar = ar.replace(bad, good)
    ar = re.sub(r"(?<![\d.,])(\d[\d,.]*) دولار(?:اً|ا)(?!\w)", lambda m: m.group(0) if m.group(1) in _FEW else m.group(1) + " دولار", ar)
    ar = re.sub(r"\b([بلو]) (?=\d)", r"\1", ar)
    ar = re.sub(r"\s+([.,!?؟،:])", r"\1", ar)
    return re.sub(r"\s{2,}", " ", ar).strip()


def _purpose(ar: str) -> str:
    """ "من أجل دخول السيارة" -> "لدخول السيارة": the tail of a key prompt."""
    ar = _END.sub("", ar.strip())
    for lead in _ORDER:
        if ar.startswith(lead + " "):
            ar = ar[len(lead) + 1 :]
            return ("لل" + ar[2:]) if ar.startswith("ال") else "ل" + ar
    return ar if ar.startswith("ل") else ""


def degenerate(ar: str, en: str) -> bool:
    """Output that is not a translation: empty, a loop, or many times longer than its source."""
    return not ar.strip() or bool(_REPEAT.search(ar)) or len(ar) > 8 * max(len(en), 4) + 20


def finish(job: Job, out: str) -> str | None:
    """The Arabic for ``job`` from the model's ``out``; ``None`` when a stand-in was lost or the output is no
    translation (the caller asks again without stand-ins)."""
    end = _punct(job.text)
    if job.fixed:
        return _END.sub("", job.fixed) + end
    if not job.src:
        return ""
    if degenerate(out, job.text):
        return None
    ar = _restore(out, job.slots)
    if ar is None:
        return None
    ar = _clean(ar, job)
    if job.action:
        tail = _purpose(ar)
        return f"{job.head} {tail}{end}" if tail else None
    ar = _END.sub("", ar)
    if job.np:
        ar = f"{job.head} {ar}"
    return ar + end
