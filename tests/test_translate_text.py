"""Game text around the translation model (streamsafe/translate/gamefix.py, ``GameTranslator``): stub model, no network.

The stub plays the part of opus-mt where it matters: it records what it was asked and answers from a table, so these
tests pin what the model is given (stand-ins, sentence case) and what comes out of the Arabic (restored, cleaned)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from streamsafe.translate import gamefix
from streamsafe.translate.translator import CachedTranslator, GameTranslator

DATA = Path(__file__).parent / "data"


class Model:
    """Answers from ``table`` (exact source -> Arabic); anything else is echoed as ``<ar>`` + the source."""

    def __init__(self, table: dict[str, str] | None = None) -> None:
        self.table = table or {}
        self.asked: list[str] = []

    def translate(self, texts: list[str]) -> list[str]:
        self.asked += texts
        return [self.table.get(t, "<ar>" + t) for t in texts]


def game(table: dict[str, str] | None = None) -> tuple[GameTranslator, Model]:
    model = Model(table)
    return GameTranslator(model), model


def one(table: dict[str, str], text: str) -> str:
    return game(table)[0].translate([text])[0]


def test_data_sets_are_well_formed():
    for name, n in (("translate_gta.tsv", 150), ("translate_gta_heldout.tsv", 50)):
        with (DATA / name).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) >= n and all(r["en"].strip() and r["ar"].strip() and r["category"] for r in rows), name
        assert all(any("؀" <= c <= "ۿ" for c in r["ar"]) for r in rows), name
        assert len({r["en"] for r in rows}) == len(rows), "no line twice"
    main = {r["en"] for r in csv.DictReader((DATA / "translate_gta.tsv").open(encoding="utf-8"), delimiter="\t")}
    held = {r["en"] for r in csv.DictReader((DATA / "translate_gta_heldout.tsv").open(encoding="utf-8"), delimiter="\t")}
    assert not main & held, "the held-out set shares no line with the set the rules were tuned on"


# ------------------------------------------------------------------------------------------------- key prompts
@pytest.mark.parametrize(
    ("en", "asked", "answer", "want"),
    [
        ("Press E to open the trunk.", "In order to open the trunk.", "من أجل فتح صندوق السيارة", "اضغط E لفتح صندوق السيارة."),
        ("Hold SHIFT to sprint.", "In order to sprint.", "من أجل الركض.", "اضغط مع الاستمرار على SHIFT للركض."),
        ("PRESS ENTER TO CONTINUE", None, None, "اضغط ENTER للمتابعة"),
        ("Press Enter To Open The Map", "In order to open the map.", "من أجل فتح الخريطة", "اضغط ENTER لفتح الخريطة"),
        ("Hold ~INPUT~ to aim.", None, None, "اضغط مع الاستمرار على ~INPUT~ للتصويب."),
        ("Press W, A, S, D to move.", "In order to move.", "من أجل التحرك", "اضغط W و A و S و D للتحرك."),
        ("Press E or F to jump.", "In order to jump.", "من أجل القفز", "اضغط E أو F للقفز."),
        ("Hold the left mouse button to fire.", None, None, "اضغط مع الاستمرار على زر الماوس الأيسر لإطلاق النار."),
        ("Press [E] to buy.", "In order to buy.", "من أجل الشراء", "اضغط E للشراء."),
        ("Press Q to take cover.", None, None, "اضغط Q للاحتماء."),
        ("Press E", None, None, "اضغط E"),
        ("Press CTRL+F to search.", "In order to search.", "من أجل البحث", "اضغط CTRL+F للبحث."),
    ],
)
def test_key_prompts_keep_the_key_and_translate_only_the_action(en, asked, answer, want):
    tr, model = game({asked: answer} if asked else {})
    assert tr.translate([en]) == [want]
    assert model.asked == ([asked] if asked else []), "a stock action needs no model call"


def test_a_prompt_the_model_cannot_turn_into_a_purpose_is_asked_again_as_a_plain_sentence():
    tr, model = game({"In order to dance.": "ارقص الآن", "Press E to dance.": "اضغط على إي للرقص."})
    assert tr.translate(["Press E to dance."]) == ["اضغط على إي للرقص."]
    assert model.asked == ["In order to dance.", "Press E to dance."]


# ------------------------------------------------------------------------------------------------- fixed lines
def test_stock_lines_are_answered_without_the_model():
    tr, model = game()
    out = tr.translate(["WASTED", "Mission failed.", "Settings", "You good?", "Loading...", "Hello there.", "Mission passed!"])
    assert out[:3] == ["قُتلت", "فشلت المهمة.", "الإعدادات"]
    assert out[3] == "هل أنت بخير؟" and out[4] == "جارٍ التحميل..." and out[6] == "نجحت المهمة!"
    assert model.asked == ["Hello there."], "only the line that is not stock"


def test_without_the_glossary_nothing_is_fixed_or_templated():
    model = Model()
    GameTranslator(model, glossary=False).translate(["WASTED", "Press E to jump."])
    assert model.asked == ["Wasted", "Press E to jump."]


# ------------------------------------------------------------------------------------------------- case
def test_all_caps_and_title_case_reach_the_model_as_sentences():
    tr, model = game()
    tr.translate(["GO TO THE BANK", "Go To The Airport", "Follow Lucia To The Marina", "Hello there, Jason"])
    assert model.asked[0] == "Please go to the bank"
    assert model.asked[1] == "Please go to the airport"
    assert model.asked[2] == "Please follow Lucia to the marina", "a name keeps its capital"
    assert model.asked[3] == "Hello there, Jason", "a normal sentence is left alone"


def test_sentence_case_keeps_acronyms_names_and_the_pronoun():
    assert gamefix.sentence_case("USE THE ATM") == "Use the ATM"
    assert gamefix.sentence_case("I'M NOT GOING TO LUCIA'S PLACE").startswith("I'm not")
    common = {"follow", "to", "the", "marina"}.__contains__
    assert gamefix.sentence_case("Follow Boobie To The Marina", common) == "Follow Boobie to the marina"
    assert gamefix.sentence_case("Press E to enter", None) == "Press E to enter"
    assert not gamefix.titled("Vice") and not gamefix.shouting("Ok")


# ------------------------------------------------------------------------------------------------- numbers and names
@pytest.mark.parametrize(
    ("en", "src", "ar", "want"),
    [
        ("You earned $2,500.", "You earned 77 dollars.", "لقد كسبت 77 دولارا.", "لقد كسبت 2,500 دولار."),
        ("Cash: $15,000", "Cash: 77 dollars", "النقدية: 000 77 دولار", None),
        (
            "The heist is worth $2.5 million.",
            "The heist is worth 77 million dollars.",
            "السرقة تساوي ٧٧ مليون دولار",
            "السرقة تساوي 2.5 مليون دولار.",
        ),
        ("Time left: 02:30", "Time left: 77", "الوقت المتبقي: 77", "الوقت المتبقي: 02:30"),
        ("Health 100%", "Health 77", "الصحة 77٪", "الصحة 100%"),
        ("Distance: 450 m", "Distance: 77", "المسافة: 77", "المسافة: 450 م"),
        ("Speed: 120 km/h", "Speed: 77", "السرعة: 77", "السرعة: 120 كم/س"),
        ("You have 5,000 gold", "You have 77 gold", "لديك 77 ذهب", "لديك 5,000 ذهب"),
        ("Mission 12 of 30", "Mission 12 of 30", "المهمة 12 من 30", "المهمة 12 من 30"),
    ],
)
def test_amounts_times_and_units_never_pass_through_the_model(en, src, ar, want):
    tr, model = game({src: ar})
    got = tr.translate([en])[0]
    assert model.asked[0] == src
    if want is None:  # the model garbled the stand-in's neighbours; the number itself is restored exactly
        assert "15,000" in got
    else:
        assert got == want


def test_several_numbers_in_one_sentence_each_get_their_own_stand_in_and_their_own_value():
    en = "Pay $1,000 and $2,000 for 100% at 12:30 on a 450 m route, 3 km, 5,000 gold."
    plan = gamefix.plan(en)
    stands = [s for s, _ in plan.slots]
    assert len(stands) == len(set(stands)) == 7 and all(s in plan.src for s in stands)
    out = gamefix.finish(plan, plan.src.replace("dollars", "دولار"))
    for value in ("1,000", "2,000", "100%", "12:30", "450 م", "3 كم", "5,000"):
        assert value in out, (value, out)


def test_a_stand_in_is_never_a_number_that_is_already_in_the_text():
    plan = gamefix.plan("Meet at 77 and pay $2,500.")
    assert plan.slots[0][0] not in ("77",) and "77 dollars" not in plan.src


def test_place_names_are_swapped_for_a_city_and_restored_whole():
    tr, model = game({"Welcome to Boston.": "مرحبا بك في بوسطن."})
    assert tr.translate(["Welcome to the Leonida Keys."]) == ["مرحبًا بك في ليونيدا كيز."]
    assert model.asked == ["Welcome to Boston."]


def test_place_names_and_a_lost_stand_in_fall_back_to_the_plain_sentence():
    tr, model = game({"Welcome to Boston.": "مرحبا بكم", "Welcome to Vice City.": "مرحبا بك في المدينة"})
    assert tr.translate(["Welcome to Vice City."]) == ["مرحبًا بك في المدينة."]
    assert model.asked == ["Welcome to Boston.", "Welcome to Vice City."]


def test_every_name_has_a_stand_in_with_a_fixed_arabic_spelling_and_the_longest_name_wins():
    assert gamefix.NAMES["rockstar games"] != gamefix.NAMES["rockstar"]
    plan = gamefix.plan("Welcome to Vice City and the Leonida Keys, Rockstar Games.")
    assert [f for _, f in plan.slots] == ["فايس سيتي", "ليونيدا كيز", "روكستار جيمز"]
    assert gamefix.plan("Vice Beach").slots[0][1] == "شاطئ فايس"
    plan = gamefix.plan("Boston, Chicago, Houston, Dallas, Seattle, Phoenix and Vice City")
    assert plan.slots == [] or plan.slots[0][0] not in plan.text  # no stand-in that is already in the line


# ------------------------------------------------------------------------------------------------- the Arabic
def test_imperatives_are_masculine_singular_and_the_please_is_dropped():
    tr, _ = game(
        {
            "Please get in the car.": "من فضلك اركبي السيارة",
            "Please take cover and wait for backup.": "رجاءً احتموا وانتظروا الدعم",
            "Please escape the police helicopter.": "رجاءً أهرب من مروحية الشرطة",
            "Please return to the safehouse.": "رجاءً عودي إلى المنزل الآمن",
            "Please meet me outside.": "من فضلك قابلني في الخارج",
            "Jason, please listen to me.": "جيسون، من فضلك اسمعني",
        }
    )
    out = tr.translate(["Get in the car.", "Take cover and wait for backup.", "Escape the police helicopter.", "Return to the safehouse."])
    assert out == ["اركب السيارة.", "احتم وانتظر الدعم.", "اهرب من مروحية الشرطة.", "عد إلى المخبأ."]
    assert tr.translate(["Meet me outside."]) == ["قابلني في الخارج."], "the pronoun 'me' is not a feminine ending"
    assert tr.translate(["Jason, listen to me."]) == ["جيسون، اسمعني."]


def test_non_imperatives_are_not_touched():
    tr, model = game({"They are waiting for you.": "إنهم ينتظرونك", "She is here.": "هي هنا"})
    assert tr.translate(["They are waiting for you.", "She is here."]) == ["إنهم ينتظرونك.", "هي هنا."]
    assert model.asked == ["They are waiting for you.", "She is here."]


def test_brackets_quotes_marks_and_closing_punctuation_are_repaired():
    tr, _ = game({"Please call Lucia, now!": 'اتصل بـ(لوسيا) "الآن"', "Where is he?": "أين هو"})
    assert tr.translate(["Where is he?"]) == ["أين هو؟"]
    assert tr.translate(["Call Lucia, now!"]) == ["اتصل بلوسيا الآن!"]
    assert one({"Hello there.": "ُمَرْحَباً"}, "Hello there.") == "مرحبا."
    tr2, _ = game({"Hello (Jason).": "مرحبا (جيسون)"})
    assert tr2.translate(["Hello (Jason)."]) == ["مرحبا (جيسون)."], "brackets that were in the English stay"
    assert one({"Please meet Jason.": "من فضلك قابل (جيسون)"}, "Meet Jason.") == "قابل جيسون."


def test_terms_use_the_game_vocabulary_not_the_models():
    tr, _ = game(
        {
            "Your wanted level has increased.": "لقد زاد مستوى مطلوبك",
            "Please go to the bank.": "اذهب إلى المصرف",
            "Waypoint ahead.": "نقطة الطريق أمامك",
            "The mission begins.": "البعثة تبدأ",
        }
    )
    out = tr.translate(["Your wanted level has increased.", "Waypoint ahead.", "The mission begins."])
    assert out == ["لقد زاد مستوى الملاحقة.", "نقطة الوجهة أمامك.", "المهمة تبدأ."]
    assert tr.translate(["Go to the bank."]) == ["اذهب إلى البنك."]


def test_slang_is_made_plain_before_the_model_and_a_noun_phrase_template_fixes_lose_the_cops():
    tr, model = game({"What's up, friend?": "ما الأمر", "The cops.": "رجال الشرطة"})
    tr.translate(["What's up, homie?", "We gotta go."])
    assert model.asked[0] == "What's up, friend?" and model.asked[1] == "We have to go."
    assert tr.translate(["Lose the cops."]) == ["اهرب من الشرطة."]
    assert gamefix.plan("Lose them!").fixed == "اهرب منهم"


def test_a_loop_or_an_empty_answer_is_never_shown():
    tr, _ = game({"Please run!": "اركض " * 20, "Run!": "اركض " * 20, "Hello there.": ""})
    assert tr.translate(["Hello there."]) == [""] and tr.translate(["Run!"]) == [""]
    assert gamefix.degenerate("مرحبا مرحبا مرحبا مرحبا مرحبا", "Hello") and gamefix.degenerate("ا" * 500, "Hi there")


def test_long_text_is_translated_sentence_by_sentence_and_joined():
    tr, model = game()
    out = tr.translate(["Stop! Drop the gun. Now!"])
    assert out[0].count("<ar>") == 3 and len(model.asked) == 3
    assert tr.translate([""]) == [""] and tr.translate([]) == []


def test_game_translator_is_a_translator_the_cache_can_wrap():
    cached = CachedTranslator(GameTranslator(Model({"Hello there.": "مرحبا"})))
    assert cached.translate(["Hello there."]) == ["مرحبا."]
    assert cached.translate(["hello there"]) == ["مرحبا."] and cached.hits == 1


def test_a_batch_costs_one_model_call_for_all_its_sentences():
    calls = []

    class Counting(Model):
        def translate(self, texts):
            calls.append(len(texts))
            return super().translate(texts)

    table = {
        "In order to jump.": "من أجل القفز",
        "Please rob the store.": "اسرق المتجر",
        "Welcome to Boston.": "مرحبا بك في بوسطن",
        "Please go inside.": "ادخل",
    }
    tr = GameTranslator(Counting(table))
    out = tr.translate(["Press E to jump.", "Rob the store.", "Mission failed.", "Welcome to Vice City. Go inside."])
    assert out == ["اضغط E للقفز.", "اسرق المتجر.", "فشلت المهمة.", "مرحبًا بك في فايس سيتي. ادخل."]
    assert calls == [4], "fixed lines need no call; the rest go in one batch"


def test_pipeline_uses_the_game_rules_by_default(tmp_path, monkeypatch):
    import inspect

    from streamsafe.translate import pipeline

    assert "GameTranslator" in inspect.getsource(pipeline.load_pipeline)
