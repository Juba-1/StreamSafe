"""Builds the static StreamSafe site into public/ (English + Arabic in every page, switched by lang.js).

Edit texts, prices or the email here, run `python build.py`, commit public/. Vercel serves public/ as is.
"""

import pathlib
import shutil

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "public"
EMAIL = "streamsafe1@gmail.com"
UPDATED_EN, UPDATED_AR = "17 November 2026", "17 نوفمبر 2026"
BASIC, PRO = 49, 99  # SAR per month
YEAR_OFF = 0.15  # yearly plan: 15% less than 12 months
BASIC_Y, PRO_Y = int(BASIC * 12 * (1 - YEAR_OFF)), int(PRO * 12 * (1 - YEAR_OFF))  # 499, 1009
TRIAL_DAYS = 3
BUY_URL = "#pricing"  # becomes the Dodo checkout link once the products exist
HAS_VIDEO = (ROOT / "media" / "demo.mp4").is_file()
FONTS = "https://fonts.googleapis.com/css2?family=Readex+Pro:wght@400;500;600&family=Reem+Kufi:wght@500;600&display=swap"


def b(en, ar, tag="div", cls=""):
    """The same content in both languages; only the active one shows."""
    c = f' class="{cls}"' if cls else ""
    return f'<{tag}{c} data-l="en">{en}</{tag}><{tag}{c} data-l="ar" dir="rtl">{ar}</{tag}>'


def nav_links(en_ar):
    en = " ".join(f'<a href="{h}">{e}</a>' for h, e, _ in en_ar)
    ar = " ".join(f'<a href="{h}">{a}</a>' for h, _, a in en_ar)
    return en, ar


TOP = [
    ("/#see", "See it work", "شوف النتيجة"),
    ("/#how", "How it works", "طريقة العمل"),
    ("/#pricing", "Pricing", "الأسعار"),
    ("/#faq", "Questions", "الأسئلة"),
]
FOOT = [
    ("/#pricing", "Pricing", "الأسعار"),
    ("/terms.html", "Terms of Service", "شروط الاستخدام"),
    ("/privacy.html", "Privacy Policy", "سياسة الخصوصية"),
    ("/refund.html", "Trial, Cancellation & Refunds", "التجربة والإلغاء والاسترداد"),
    ("/contact.html", "Contact", "تواصل معنا"),
]


def page(title, desc, body):
    ten, tar = nav_links(TOP)
    fen, far = nav_links(FOOT)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:image" content="/media/roof.webp">
<meta name="theme-color" content="#120b2b">
<link rel="icon" href="/favicon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="/style.css">
<script src="/lang.js"></script>
</head>
<body>
<a class="skip" href="#main">{b("Skip to content", "انتقل للمحتوى", "span")}</a>
<header class="top"><div class="wrap">
<a class="brand" href="/"><img src="/logo.webp" alt="" width="42" height="42">StreamSafe</a>
<nav class="nav" data-l="en" aria-label="Main">{ten}</nav><nav class="nav" data-l="ar" aria-label="الرئيسية">{tar}</nav>
<button class="lang" type="button">{b("العربية", "English", "span")}</button>
<a class="btn btn-main" href="{BUY_URL}">{b("Start free trial", "ابدأ التجربة المجانية", "span")}</a>
</div></header>
<main id="main">
{body}
</main>
<footer class="foot"><div class="wrap">
<div><a class="brand" href="/"><img src="/logo.webp" alt="" width="42" height="42">StreamSafe</a>
<p style="margin-top:12px">{
        b(f'Questions: <a href="mailto:{EMAIL}">{EMAIL}</a>', f'للتواصل: <a href="mailto:{EMAIL}">{EMAIL}</a>', "span")
    }</p></div>
<nav class="links" data-l="en" aria-label="Legal">{fen}</nav><nav class="links" data-l="ar" aria-label="روابط">{far}</nav>
<p class="legal">{
        b(
            "© 2026 StreamSafe. StreamSafe is an independent product and is not affiliated with, endorsed or sponsored by Rockstar Games or Take-Two Interactive. Grand Theft Auto and GTA are their trademarks. Payments are processed by Dodo Payments, our Merchant of Record.",
            "© 2026 StreamSafe. منتج مستقل، وليس تابعًا لشركة Rockstar Games أو Take-Two Interactive ولا معتمدًا منهما. Grand Theft Auto و GTA علامات تجارية مملوكة لهما. المدفوعات تتم عن طريق Dodo Payments بصفتها البائع المسجّل.",
            "span",
        )
    }</p>
</div></footer>
</body>
</html>
"""


# ---------- home ----------

SHOTS = [
    ("roof", "Rooftop pool: every swimsuit covered, the city stays visible.", "مسبح على السطح: كل ملابس السباحة متغطية والمدينة باينة."),
    ("night", "Night drive: covered while she moves.", "قيادة بالليل: التغطية تمشي معها وهي تتحرك."),
    ("club", "Club: a whole crowd, each person covered.", "نادي ليلي: زحمة كاملة وكل شخص متغطي."),
    ("store", "Store robbery: only what needs covering.", "سرقة محل: يغطي اللي يحتاج تغطية بس."),
]

shots_html = "".join(
    f'<figure><img src="/media/{n}.webp" alt="" width="960" height="540" loading="lazy">{b(en, ar, "figcaption")}</figure>'
    for n, en, ar in SHOTS
)

SCREEN = (
    """<video id="hero-video" autoplay muted loop playsinline preload="auto" poster="/media/poster.webp" width="1280" height="720"
 aria-label="GTA 6 trailer after StreamSafe: swimsuits and outfits covered with a navy shirt carrying the StreamSafe logo"><source src="/media/demo.mp4" type="video/mp4"></video>
<button class="pp" type="button" aria-pressed="false"><span data-l="en">Pause</span><span data-l="ar">إيقاف</span></button>"""
    if HAS_VIDEO
    else """<img src="/media/roof.webp" alt="GTA 6 trailer scene after StreamSafe: swimsuits covered with a navy shirt carrying the StreamSafe logo" width="960" height="540" fetchpriority="high">
<img src="/media/night.webp" alt="" width="960" height="540">
<img src="/media/club.webp" alt="" width="960" height="540">
<img src="/media/store.webp" alt="" width="960" height="540">"""
)


HOME = f"""
<section class="hero"><div class="wrap">
<div>
{b("<h1>Stream GTA 6. Keep it clean.</h1>", "<h1>بث GTA 6 بدون إحراج</h1>")}
{
    b(
        '<p class="lead">StreamSafe sees your game a couple of seconds before your viewers do. When a revealing outfit or scene shows up, it puts a cover on it first, so your stream stays family-friendly and inside platform rules.</p>',
        '<p class="lead">StreamSafe يشوف اللعب قبل المشاهدين بثانيتين. إذا طلعت شخصية لابسة لبس غير محتشم أو مشهد غير لائق، يغطيه قبل ما يوصل للبث، عشان يبقى بثك مناسب لكل العائلة وضمن قوانين المنصات.</p>',
    )
}
<div class="ctas">
<a class="btn btn-main" href="{BUY_URL}">{b(f"Try it free for {TRIAL_DAYS} days", f"جرّبه مجانًا {TRIAL_DAYS} أيام", "span")}</a>
<a class="btn btn-line" href="#see">{b("See real output", "شوف نتيجة حقيقية", "span")}</a>
</div>
{
    b(
        '<p class="fine">Windows app. Works with OBS on Twitch, Kick, YouTube and more. PC, PlayStation and Xbox.</p>',
        '<p class="fine">برنامج ويندوز يشتغل مع OBS على Twitch و Kick و YouTube وغيرها. بي سي وبلايستيشن وإكس بوكس.</p>',
    )
}
</div>
<figure class="monitor">
<div class="screen">
{SCREEN}
<div class="live"><b>LIVE</b><span>StreamSafe</span></div>
</div>
<div class="lane"><span>{b("Your game", "لعبتك", "span")}</span><span class="bar"><i></i></span><span>{
    b("Viewers +2.5 s", "المشاهدين ‎+2.5 ث", "span")
}</span></div>
{
    b(
        "<figcaption>What viewers get: real StreamSafe output on the official GTA 6 trailer.</figcaption>",
        "<figcaption>هذا اللي يشوفه المشاهد: مخرجات StreamSafe الحقيقية على العرض الرسمي للعبة.</figcaption>",
    )
}
</figure>
</div></section>

<section class="band band-cover" id="see"><div class="wrap">
<div class="head">
{
    b(
        "<h2>Real output, not a mock-up</h2><p>These frames come straight out of StreamSafe, run on Rockstar's official GTA 6 trailer. Nothing was edited by hand.</p>",
        "<h2>نتيجة حقيقية مو تصميم</h2><p>هذي اللقطات طالعة مباشرة من StreamSafe بعد تشغيله على العرض الرسمي للعبة من Rockstar، بدون أي تعديل يدوي.</p>",
    )
}
</div>
<div class="shots">{shots_html}</div>
{
    b(
        '<p class="source">Source footage: Grand Theft Auto VI Trailer 1 © Rockstar Games, shown processed to demonstrate the software.</p>',
        '<p class="source">المقطع الأصلي: العرض الأول للعبة Grand Theft Auto VI © Rockstar Games، معروض بعد المعالجة لتوضيح عمل البرنامج.</p>',
    )
}
</div></section>

<section class="band"><div class="wrap">
<div class="head">
{
    b(
        "<h2>Three ways it covers</h2><p>StreamSafe picks the lightest cover that is still safe, and goes bigger whenever it is unsure.</p>",
        "<h2>ثلاث طرق للتغطية</h2><p>StreamSafe يختار أخف تغطية تكون آمنة، ويكبّرها إذا ما كان متأكد.</p>",
    )
}
</div>
<div class="ways">
<div class="way"><img src="/media/night.webp" alt="" width="960" height="540" loading="lazy">
{
    b(
        "<h3>Cloth</h3><p>A loose navy shirt with our logo is drawn over the person and follows them as they move. The rest of the scene stays as it is, so your viewers can still follow the game.</p>",
        "<h3>القماش</h3><p>قميص كحلي واسع عليه شعارنا يتركب على الشخص ويمشي معه وهو يتحرك. باقي المشهد يبقى مثل ما هو، فالمشاهد يقدر يتابع اللعب.</p>",
    )
}</div>
<div class="way"><img src="/media/scene.webp" alt="" width="960" height="640" loading="lazy">
{
    b(
        "<h3>Full-scene cover</h3><p>Some scenes, like strip clubs and private dances, are covered as a whole with a full-screen card until the scene ends. Game audio is muted with it.</p>",
        "<h3>تغطية المشهد كامل</h3><p>بعض المشاهد مثل نوادي التعري والرقص الخاص تتغطى كاملة بشاشة لين ينتهي المشهد، وصوت اللعبة ينكتم معها.</p>",
    )
}</div>
<div class="way"><img src="/media/club.webp" alt="" width="960" height="540" loading="lazy">
{
    b(
        "<h3>Safety net</h3><p>If the app falls behind, loses the game feed or hits an error, it covers the whole screen instead of guessing. It never lets a doubtful frame through.</p>",
        "<h3>شبكة الأمان</h3><p>إذا تأخر البرنامج أو انقطع عنه اللعب أو صار خطأ، يغطي الشاشة كاملة بدل ما يخمّن. ما يطلّع أي لقطة مشكوك فيها.</p>",
    )
}</div>
</div>
{
    b(
        "<p><strong>No blur</strong>Blur and pixels can be undone or seen through. Covers are solid.</p><p><strong>Stays on long enough</strong>A cover is released only after the person is gone for a moment, so nothing flashes at the edges.</p><p><strong>Checked on the final video</strong>We test what actually leaves your PC, not just what the app intended.</p>",
        "<p><strong>بدون تمويه</strong>التمويه والبكسلة ممكن تنشاف من خلالها. التغطية عندنا صلبة.</p><p><strong>ما تختفي بسرعة</strong>التغطية ما تنشال إلا بعد ما يطلع الشخص بلحظة، عشان ما يبان شيء على الأطراف.</p><p><strong>نفحص الفيديو النهائي</strong>نختبر الفيديو اللي يطلع فعلًا من جهازك، مو بس اللي يقصده البرنامج.</p>",
        "div",
        "rules",
    )
}
</div></section>

<section class="band band-cover" id="how"><div class="wrap">
<div class="head">
{
    b(
        "<h2>Where it sits in your setup</h2><p>OBS sends to StreamSafe instead of straight to the platform. StreamSafe holds the picture for about two seconds, covers what needs covering and passes it on.</p>",
        "<h2>وين مكانه في إعداداتك</h2><p>OBS يرسل لـ StreamSafe بدل ما يرسل للمنصة مباشرة. StreamSafe يمسك الصورة حوالي ثانيتين، يغطي اللي يحتاج تغطية، ويرسلها للمنصة.</p>",
    )
}
</div>
<div class="path" aria-hidden="true">
<div class="node"><b>OBS</b><small>game · camera · mic</small></div><span class="arrow">→</span>
<div class="node me"><b>StreamSafe</b><small>2–3 s delay · covers</small></div><span class="arrow">→</span>
<div class="node"><b>Twitch · Kick · YouTube</b><small>or a video file</small></div>
</div>
<ol class="steps">
<li>{
    b(
        "<h3>Install</h3><p>Download the Windows app and enter the licence key from your email.</p>",
        "<h3>ثبّت</h3><p>حمّل برنامج ويندوز وحط مفتاح الاشتراك اللي يوصلك على الإيميل.</p>",
    )
}</li>
<li>{
    b(
        "<h3>Pick your platform</h3><p>Choose Twitch, Kick, YouTube or another, and paste your stream key. Typos in the address are fixed for you.</p>",
        "<h3>اختر منصتك</h3><p>اختر Twitch أو Kick أو YouTube أو غيرها، والصق مفتاح البث. أخطاء الرابط الشائعة يصلحها البرنامج.</p>",
    )
}</li>
<li>{
    b(
        "<h3>Point OBS at StreamSafe</h3><p>The app shows one address to paste into OBS. Press Test connection to check everything before you go live.</p>",
        "<h3>وجّه OBS للبرنامج</h3><p>البرنامج يعطيك عنوان واحد تحطه في OBS. اضغط اختبار الاتصال عشان تتأكد قبل ما تبدأ.</p>",
    )
}</li>
<li>{
    b(
        "<h3>Go live</h3><p>Press Start in StreamSafe, then Start Streaming in OBS. Your game, camera and mic stay in sync.</p>",
        "<h3>ابدأ البث</h3><p>اضغط تشغيل في StreamSafe وبعدها ابدأ البث في OBS. اللعبة والكاميرا والمايك يبقون متزامنين.</p>",
    )
}</li>
</ol>
</div></section>

<section class="band band-sand"><div class="wrap">
<div class="head">{b("<h2>The details</h2><p>What you need and what you get.</p>", "<h2>التفاصيل</h2><p>وش تحتاج ووش تحصل.</p>")}</div>
{
    b(
        '''<dl><div><dt>Platforms</dt><dd>Twitch, Kick, YouTube, Facebook, X, Trovo and any RTMP or RTMPS server</dd></div>
<div><dt>Not streaming live?</dt><dd>Record a covered MP4 or MKV and upload it to YouTube later</dd></div>
<div><dt>Picture</dt><dd>Keeps your resolution and frame rate, up to 1440p and 60 fps, at a full bitrate</dd></div>
<div><dt>Delay</dt><dd>2 or 3 seconds, set automatically for your PC; you can choose it yourself</dd></div></dl>
<dl><div><dt>System</dt><dd>Windows 10 or 11 (64-bit) with OBS Studio</dd></div>
<div><dt>Graphics card</dt><dd>NVIDIA GeForce RTX recommended. Tested on RTX 3070, RTX 3080 and an RTX 4060 laptop</dd></div>
<div><dt>Consoles</dt><dd>PlayStation 5 and Xbox through a capture card into OBS</dd></div>
<div><dt>Privacy</dt><dd>Video is processed on your PC and never uploaded to us</dd></div></dl>''',
        '''<dl><div><dt>المنصات</dt><dd>Twitch و Kick و YouTube و Facebook و X و Trovo وأي سيرفر RTMP أو RTMPS</dd></div>
<div><dt>ما تبث مباشر؟</dt><dd>سجّل فيديو MP4 أو MKV متغطي وارفعه على يوتيوب بعدين</dd></div>
<div><dt>الصورة</dt><dd>نفس الدقة وعدد الإطارات، لين 1440p و 60 إطار، وبجودة كاملة</dd></div>
<div><dt>التأخير</dt><dd>ثانيتين أو ثلاث، يختارها البرنامج حسب جهازك، وتقدر تغيرها بنفسك</dd></div></dl>
<dl><div><dt>النظام</dt><dd>ويندوز 10 أو 11 (64 بت) مع OBS Studio</dd></div>
<div><dt>كرت الشاشة</dt><dd>يُنصح بـ NVIDIA GeForce RTX. جربناه على RTX 3070 و RTX 3080 ولابتوب RTX 4060</dd></div>
<div><dt>الأجهزة المنزلية</dt><dd>بلايستيشن 5 وإكس بوكس عن طريق كارت تصوير مع OBS</dd></div>
<div><dt>الخصوصية</dt><dd>الفيديو يتعالج على جهازك وما ينرفع لنا أبدًا</dd></div></dl>''',
        "div",
        "specs",
    )
}
</div></section>

<section class="band band-sand" id="pricing" style="padding-top:0"><div class="wrap">
<div class="head">{
    b(
        f"<h2>Pricing</h2><p>Every plan starts with a {TRIAL_DAYS}-day free trial. Cancel before it ends and you pay nothing.</p>",
        f"<h2>الأسعار</h2><p>كل باقة تبدأ بتجربة مجانية {TRIAL_DAYS} أيام. إذا ألغيت قبل ما تخلص ما تدفع شيء.</p>",
    )
}</div>
<div class="period" role="radiogroup">
<input type="radio" name="period" id="pm" checked><label for="pm">{b("Monthly", "شهري", "span")}</label>
<input type="radio" name="period" id="py"><label for="py">{b("Yearly, save 15%", "سنوي، وفّر 15%", "span")}</label>
</div>
<div class="plans">
<div class="plan">
{
    b(
        f'<h3>Basic</h3><p class="tag">Everything you need to stream safely</p><p class="price"><span class="m">{BASIC} <small>SAR / month</small></span><span class="y">{BASIC_Y:,} <small>SAR / year</small></span></p><p class="per"><span class="m">after the {TRIAL_DAYS}-day free trial</span><span class="y">{BASIC * 12 - BASIC_Y:,} SAR less than 12 monthly payments. Starts after the {TRIAL_DAYS}-day trial</span></p><ul><li>Cloth, full-scene and safety-net covers</li><li>Live streaming and recording</li><li>All platforms and any RTMP server</li><li>Full resolution and frame rate</li><li>StreamSafe cover screen</li><li>One PC</li></ul>',
        f'<h3>الأساسية</h3><p class="tag">كل اللي تحتاجه عشان تبث بأمان</p><p class="price"><span class="m">{BASIC} <small>ريال / شهريًا</small></span><span class="y">{BASIC_Y:,} <small>ريال / سنويًا</small></span></p><p class="per"><span class="m">بعد التجربة المجانية {TRIAL_DAYS} أيام</span><span class="y">توفر {BASIC * 12 - BASIC_Y:,} ريال عن 12 دفعة شهرية، ويبدأ بعد التجربة {TRIAL_DAYS} أيام</span></p><ul><li>تغطية بالقماش وتغطية المشهد وشبكة الأمان</li><li>بث مباشر وتسجيل</li><li>كل المنصات وأي سيرفر RTMP</li><li>نفس الدقة وعدد الإطارات</li><li>شاشة التغطية بشعار StreamSafe</li><li>جهاز واحد</li></ul>',
    )
}
<a class="btn btn-main" href="{BUY_URL}">{b("Start free trial", "ابدأ التجربة", "span")}</a>
</div>
<div class="plan pro">
{
    b(
        f'<h3>Pro</h3><p class="tag">For creators who want their own brand on screen</p><p class="price"><span class="m">{PRO} <small>SAR / month</small></span><span class="y">{PRO_Y:,} <small>SAR / year</small></span></p><p class="per"><span class="m">after the {TRIAL_DAYS}-day free trial</span><span class="y">{PRO * 12 - PRO_Y:,} SAR less than 12 monthly payments. Starts after the {TRIAL_DAYS}-day trial</span></p><ul><li>Everything in Basic</li><li>Your own logo on the cover screen and the cloth</li><li>Priority support by email</li><li>One PC</li></ul>',
        f'<h3>برو</h3><p class="tag">لصنّاع المحتوى اللي يبون شعارهم على الشاشة</p><p class="price"><span class="m">{PRO} <small>ريال / شهريًا</small></span><span class="y">{PRO_Y:,} <small>ريال / سنويًا</small></span></p><p class="per"><span class="m">بعد التجربة المجانية {TRIAL_DAYS} أيام</span><span class="y">توفر {PRO * 12 - PRO_Y:,} ريال عن 12 دفعة شهرية، ويبدأ بعد التجربة {TRIAL_DAYS} أيام</span></p><ul><li>كل مزايا الأساسية</li><li>شعارك الخاص على شاشة التغطية والقماش</li><li>دعم بأولوية على الإيميل</li><li>جهاز واحد</li></ul>',
    )
}
<a class="btn btn-main" href="{BUY_URL}">{b("Start free trial", "ابدأ التجربة", "span")}</a>
</div>
</div>
{
    b(
        f"<p><strong>Card needed to start</strong>The trial asks for a payment card. Nothing is charged during the {TRIAL_DAYS} days.</p><p><strong>Then monthly</strong>On day {TRIAL_DAYS + 1} your plan starts and renews every month until you cancel.</p><p><strong>Cancel any time</strong>From the link in your email. One free trial per person, card and PC.</p>",
        f"<p><strong>تحتاج بطاقة للبدء</strong>التجربة تطلب بطاقة دفع، وما ينخصم شيء خلال الـ {TRIAL_DAYS} أيام.</p><p><strong>بعدها شهري</strong>في اليوم {TRIAL_DAYS + 1} يبدأ اشتراكك ويتجدد كل شهر لين تلغيه.</p><p><strong>تلغي متى ما تبي</strong>من الرابط اللي في إيميلك. تجربة مجانية وحدة لكل شخص وبطاقة وجهاز.</p>",
        "div",
        "trial",
    )
}
{
    b(
        '<p class="source" style="color:#5c4e6e">Prices in Saudi riyals; your bank may show your local currency. Taxes may be added at checkout depending on your country. See <a href="/refund.html">Trial, Cancellation &amp; Refunds</a>.</p>',
        '<p class="source" style="color:#5c4e6e">الأسعار بالريال السعودي وقد يظهر المبلغ بعملتك المحلية. قد تُضاف ضرائب حسب بلدك عند الدفع. اطلع على <a href="/refund.html">التجربة والإلغاء والاسترداد</a>.</p>',
    )
}
</div></section>
"""

FAQ = [
    (
        "Can it get me banned?",
        "No tool can promise that, and we don't. StreamSafe is built to cover more rather than less and covers the whole screen when it is unsure. You still decide what you play and remain responsible for your stream.",
        "هل ممكن يسبب لي حظر؟",
        "ما فيه برنامج يقدر يضمن هذا ولا نقول إننا نضمنه. StreamSafe مصمم يغطي أكثر لا أقل، ويغطي الشاشة كاملة إذا ما كان متأكد. ويبقى قرار اللعب ومسؤولية البث عليك.",
    ),
    (
        "I play on PlayStation. Does it work?",
        "Yes. Connect the console to a capture card, bring it into OBS as usual, and send OBS to StreamSafe. The app does the rest on your PC.",
        "ألعب على بلايستيشن، يشتغل معي؟",
        "إيه. وصّل الجهاز بكارت تصوير، ودخّله في OBS مثل العادة، وخلي OBS يرسل لـ StreamSafe. والباقي يسويه البرنامج على البي سي.",
    ),
    (
        "Will my stream look worse?",
        "No. StreamSafe keeps your resolution, frame rate and a full bitrate. The only change your viewers notice is a short delay and the covers.",
        "بتقل جودة البث؟",
        "لا. StreamSafe يحافظ على الدقة وعدد الإطارات والجودة كاملة. الفرق الوحيد اللي يلاحظه المشاهد هو تأخير بسيط والتغطية.",
    ),
    (
        "I don't stream live. Can I use it for YouTube videos?",
        "Yes. Choose Record instead of a platform and StreamSafe saves a covered video file you can edit and upload.",
        "ما أبث مباشر، أقدر أستخدمه لفيديوهات يوتيوب؟",
        "إيه. اختر التسجيل بدل المنصة، و StreamSafe يحفظ لك فيديو متغطي تقدر تعدّله وترفعه.",
    ),
    (
        "What if my PC is not fast enough?",
        "The app measures your PC and adjusts itself. If it still falls behind, it covers the screen rather than letting anything through, and tells you what to change.",
        "إذا جهازي ضعيف وش يصير؟",
        "البرنامج يقيس جهازك ويضبط نفسه. وإذا تأخر برضو، يغطي الشاشة بدل ما يطلّع شيء، ويقولك وش تغير.",
    ),
    (
        "Does it hear or record my voice?",
        "Your camera and microphone pass through with the game so they stay in sync. They are processed on your PC and never sent to us.",
        "هل يسمع صوتي أو يسجله؟",
        "الكاميرا والمايك يمرون مع اللعب عشان يبقون متزامنين. كل شيء يتعالج على جهازك وما يوصلنا أبدًا.",
    ),
    (
        "How do the trial and cancelling work?",
        f"You start the {TRIAL_DAYS}-day trial with a payment card. If you cancel before it ends, you are not charged. If you don't, your plan starts on day {TRIAL_DAYS + 1}. You can cancel any time from the link in your email.",
        "كيف تشتغل التجربة والإلغاء؟",
        f"تبدأ التجربة {TRIAL_DAYS} أيام ببطاقة دفع. إذا ألغيت قبل ما تخلص ما ينخصم منك شيء، وإذا ما ألغيت يبدأ اشتراكك في اليوم {TRIAL_DAYS + 1}. وتقدر تلغي أي وقت من الرابط اللي في إيميلك.",
    ),
    (
        "Is StreamSafe made by Rockstar?",
        "No. StreamSafe is an independent product made for streamers. It does not change or read the game's files; it only works on the video OBS sends it.",
        "هل StreamSafe من Rockstar؟",
        "لا. StreamSafe منتج مستقل للستريمرز. ما يعدل ملفات اللعبة ولا يقرأها، يشتغل بس على الفيديو اللي يرسله OBS.",
    ),
]

faq_html = b(
    "".join(f'<details name="faq"><summary>{qe}</summary><p class="a">{ae}</p></details>' for qe, ae, _, _ in FAQ),
    "".join(f'<details name="faq-ar"><summary>{qa}</summary><p class="a">{aa}</p></details>' for _, _, qa, aa in FAQ),
)

HOME += f"""
<section class="band band-sand" id="faq" style="padding-top:0"><div class="wrap">
<div class="head">{b("<h2>Questions streamers ask</h2>", "<h2>أسئلة يسألها الستريمرز</h2>")}</div>
<div class="faq">{faq_html}</div>
</div></section>

<section class="band close"><div class="wrap">
<img src="/logo.webp" alt="" width="150" height="150" loading="lazy">
{
    b(
        "<h2>Ready before launch day</h2><p>GTA 6 arrives on 19 November. Set StreamSafe up once and stream without worrying.</p>",
        "<h2>جاهز قبل يوم الإطلاق</h2><p>GTA 6 ينزل 19 نوفمبر. جهّز StreamSafe مرة وحدة وابث وأنت مرتاح.</p>",
    )
}
<a class="btn btn-main" href="{BUY_URL}">{b(f"Try it free for {TRIAL_DAYS} days", f"جرّبه مجانًا {TRIAL_DAYS} أيام", "span")}</a>
</div></section>
"""


# ---------- legal pages ----------


def doc(title_en, title_ar, en, ar):
    return f"""<div class="doc">
<div data-l="en"><h1>{title_en}</h1><p class="updated">Last updated: {UPDATED_EN}</p>{en}</div>
<div data-l="ar" dir="rtl"><h1>{title_ar}</h1><p class="updated">آخر تحديث: {UPDATED_AR}</p>{ar}</div>
</div>"""


MOR_EN = "Our order process is conducted by our online reseller Dodo Payments. Dodo Payments is the Merchant of Record for all our orders and handles payment, billing, taxes and invoices."
MOR_AR = "تتم عملية الشراء عن طريق بائعنا الإلكتروني Dodo Payments، وهو البائع المسجّل (Merchant of Record) لجميع الطلبات ويتولى الدفع والفوترة والضرائب والفواتير."

TERMS = doc(
    "Terms of Service",
    "شروط الاستخدام",
    f"""
<p>These terms apply to the StreamSafe Windows application, this website and StreamSafe subscriptions ("StreamSafe", "we", "us"). StreamSafe is operated by an independent developer based in Saudi Arabia. By starting a trial, buying a subscription or using the app, you agree to these terms.</p>
<h2>1. The service</h2>
<p>StreamSafe is software that runs on your own PC. It delays your stream by a few seconds and automatically covers characters and scenes it judges to be revealing in Grand Theft Auto VI gameplay, before the video is sent to your streaming platform or saved to a file.</p>
<h2>2. No promise of perfect detection</h2>
<p>StreamSafe is designed to cover more rather than less, and to cover the whole screen when it is unsure or when something fails. Even so, automatic detection cannot be perfect. You remain responsible for the content of your streams and recordings and for following the rules of the platforms you use. We are not liable for platform actions such as warnings, suspensions or bans.</p>
<h2>3. Free trial</h2>
<p>New customers get a {TRIAL_DAYS}-day free trial. A payment card is required to start it. If you do not cancel before the trial ends, your subscription starts automatically and the first monthly payment is charged. One trial per person, payment card and PC; we may refuse or end trials that are used to get repeated free access.</p>
<h2>4. Subscriptions and payment</h2>
<p>Subscriptions are billed monthly or yearly in advance, as you choose, at the price on our <a href="/#pricing">pricing section</a> and renew automatically for the same period until cancelled. {MOR_EN} If we change prices, the new price applies from your next billing period and we tell you by email beforehand.</p>
<h2>5. Licence</h2>
<p>Each subscription gives you a personal, non-transferable licence key for one PC at a time. You can move it to another PC by deactivating it in the app. You may not share, resell, copy, reverse engineer or redistribute the app or its models, or try to bypass the licence check.</p>
<h2>6. Cancellation and refunds</h2>
<p>You can cancel at any time. Details are in our <a href="/refund.html">Trial, Cancellation &amp; Refunds policy</a>.</p>
<h2>7. Acceptable use</h2>
<p>Use StreamSafe only for lawful purposes and with streaming accounts and content you are entitled to use. Do not use it to break any law or any platform's terms.</p>
<h2>8. Updates and connection</h2>
<p>We may update the app to improve detection, fix problems or support new platforms. The app needs an internet connection at least every few days to confirm your subscription.</p>
<h2>9. Liability</h2>
<p>To the extent the law allows, StreamSafe is provided "as is", and our total liability for any claim is limited to the amount you paid us in the three months before the claim.</p>
<h2>10. Trademarks</h2>
<p>StreamSafe is not affiliated with, endorsed or sponsored by Rockstar Games or Take-Two Interactive. Grand Theft Auto and GTA are their trademarks.</p>
<h2>11. Changes and contact</h2>
<p>We may update these terms and will post the new version here. Questions: <a href="mailto:{EMAIL}">{EMAIL}</a>.</p>
""",
    f"""
<p>هذه الشروط تنطبق على برنامج StreamSafe لويندوز وهذا الموقع واشتراكات StreamSafe ("StreamSafe" أو "نحن"). يدير StreamSafe مطوّر مستقل مقيم في المملكة العربية السعودية. ببدئك التجربة أو شرائك اشتراكًا أو استخدامك البرنامج فأنت توافق على هذه الشروط.</p>
<h2>1. الخدمة</h2>
<p>StreamSafe برنامج يشتغل على جهازك. يؤخر البث بضع ثوانٍ ويغطي تلقائيًا الشخصيات والمشاهد غير اللائقة في لعبة Grand Theft Auto VI قبل إرسال الفيديو لمنصة البث أو حفظه في ملف.</p>
<h2>2. لا يوجد وعد بكشف كامل</h2>
<p>StreamSafe مصمم ليغطي أكثر لا أقل، ويغطي الشاشة كاملة إذا ما كان متأكد أو إذا صار خلل. ومع ذلك فالكشف التلقائي لا يمكن أن يكون كاملًا. تبقى أنت المسؤول عن محتوى بثك وتسجيلاتك وعن الالتزام بقوانين المنصات، ولسنا مسؤولين عن إجراءات المنصات مثل التحذير أو الإيقاف أو الحظر.</p>
<h2>3. التجربة المجانية</h2>
<p>العملاء الجدد يحصلون على تجربة مجانية {TRIAL_DAYS} أيام، ويلزم إدخال بطاقة دفع لبدئها. إذا ما ألغيت قبل نهاية التجربة يبدأ اشتراكك تلقائيًا وتُخصم أول دفعة شهرية. تجربة واحدة لكل شخص وبطاقة دفع وجهاز، ويحق لنا رفض أو إنهاء التجارب المستخدمة للحصول على استخدام مجاني متكرر.</p>
<h2>4. الاشتراكات والدفع</h2>
<p>الاشتراك يُدفع مقدمًا شهريًا أو سنويًا حسب اختيارك بالسعر الظاهر في <a href="/#pricing">قسم الأسعار</a> ويتجدد تلقائيًا لنفس المدة لين تلغيه. {MOR_AR} إذا تغيرت الأسعار فالسعر الجديد يطبق من فترة الفوترة التالية ونبلغك بالإيميل قبلها.</p>
<h2>5. الترخيص</h2>
<p>كل اشتراك يعطيك مفتاح ترخيص شخصي غير قابل للتحويل لجهاز واحد في نفس الوقت، وتقدر تنقله لجهاز ثاني بإلغاء تفعيله من البرنامج. لا يجوز مشاركة البرنامج أو نماذجه أو بيعها أو نسخها أو الهندسة العكسية لها أو تجاوز فحص الاشتراك.</p>
<h2>6. الإلغاء والاسترداد</h2>
<p>تقدر تلغي في أي وقت، والتفاصيل في <a href="/refund.html">سياسة التجربة والإلغاء والاسترداد</a>.</p>
<h2>7. الاستخدام المقبول</h2>
<p>استخدم StreamSafe لأغراض مشروعة فقط، ومع حسابات بث ومحتوى يحق لك استخدامها، ولا تستخدمه لمخالفة أي نظام أو شروط أي منصة.</p>
<h2>8. التحديثات والاتصال</h2>
<p>قد نحدّث البرنامج لتحسين الكشف أو إصلاح المشاكل أو دعم منصات جديدة. يحتاج البرنامج اتصال إنترنت مرة على الأقل كل بضعة أيام للتأكد من الاشتراك.</p>
<h2>9. المسؤولية</h2>
<p>في حدود ما يسمح به النظام، يُقدَّم StreamSafe "كما هو"، ومسؤوليتنا الكاملة عن أي مطالبة محدودة بالمبلغ الذي دفعته لنا خلال الأشهر الثلاثة السابقة للمطالبة.</p>
<h2>10. العلامات التجارية</h2>
<p>StreamSafe ليس تابعًا لشركة Rockstar Games أو Take-Two Interactive ولا معتمدًا منهما. Grand Theft Auto و GTA علامات تجارية مملوكة لهما.</p>
<h2>11. التعديلات والتواصل</h2>
<p>قد نعدّل هذه الشروط وننشر النسخة الجديدة هنا. للأسئلة: <a href="mailto:{EMAIL}">{EMAIL}</a>.</p>
""",
)

PRIVACY = doc(
    "Privacy Policy",
    "سياسة الخصوصية",
    f"""
<p>This policy explains what information StreamSafe handles and why.</p>
<div class="box"><p><strong>Your video and audio stay on your PC.</strong> StreamSafe processes your gameplay, camera and microphone on your own computer. We never receive, upload, store or watch them. They go only to the platform or file you choose.</p></div>
<h2>What we process</h2>
<ul>
<li><strong>Licence check:</strong> when you activate the app, and every few hours while it runs, the app sends your licence key, an activation ID and your PC's name to Dodo Payments to confirm your subscription is active.</li>
<li><strong>Purchase details:</strong> when you start a trial or subscribe, Dodo Payments (our Merchant of Record) collects your name, email, billing country and payment details. We receive your email, plan and subscription status, never your full card details. See Dodo Payments' privacy policy at <a href="https://dodopayments.com/privacy-policy">dodopayments.com</a>.</li>
<li><strong>Updates:</strong> the app may download update and model files from our servers; like any download, this shares your IP address with the hosting provider.</li>
<li><strong>Support emails:</strong> if you email us, we keep the conversation to help you.</li>
</ul>
<h2>Stream keys</h2>
<p>Your stream key is saved only on your PC in the app's settings and is sent only to the platform you stream to.</p>
<h2>What we don't do</h2>
<p>We don't sell your data, show ads, or use tracking cookies on this website. The site remembers your language choice in your own browser.</p>
<h2>Keeping and deleting data</h2>
<p>We keep subscription records while your subscription is active and afterwards for as long as the law requires. To see, correct or delete your information, email <a href="mailto:{EMAIL}">{EMAIL}</a>.</p>
""",
    f"""
<p>هذه السياسة توضح وش المعلومات اللي يتعامل معها StreamSafe وليش.</p>
<div class="box"><p><strong>الفيديو والصوت يبقون على جهازك.</strong> StreamSafe يعالج اللعب والكاميرا والمايك على جهازك فقط. ما نستلمها ولا نرفعها ولا نحفظها ولا نشوفها أبدًا، وتروح بس للمنصة أو الملف اللي تختاره.</p></div>
<h2>وش اللي نعالجه</h2>
<ul>
<li><strong>فحص الاشتراك:</strong> عند تفعيل البرنامج وكل بضع ساعات أثناء تشغيله، يرسل البرنامج مفتاح الاشتراك ورقم التفعيل واسم جهازك إلى Dodo Payments للتأكد أن اشتراكك فعّال.</li>
<li><strong>بيانات الشراء:</strong> عند بدء التجربة أو الاشتراك تجمع Dodo Payments (البائع المسجّل) اسمك وإيميلك وبلدك وبيانات الدفع. نحن نستلم الإيميل والباقة وحالة الاشتراك فقط، وما نشوف بيانات بطاقتك كاملة. اطلع على سياسة خصوصية Dodo Payments في <a href="https://dodopayments.com/privacy-policy">dodopayments.com</a>.</li>
<li><strong>التحديثات:</strong> قد يحمّل البرنامج ملفات التحديث والنماذج من سيرفراتنا، ومثل أي تحميل يشارك هذا عنوان IP مع مزود الاستضافة.</li>
<li><strong>رسائل الدعم:</strong> إذا راسلتنا نحتفظ بالمحادثة عشان نساعدك.</li>
</ul>
<h2>مفاتيح البث</h2>
<p>مفتاح البث يُحفظ على جهازك فقط في إعدادات البرنامج، ولا يُرسل إلا للمنصة اللي تبث عليها.</p>
<h2>وش اللي ما نسويه</h2>
<p>ما نبيع بياناتك، وما نعرض إعلانات، وما نستخدم كوكيز تتبع في هذا الموقع. الموقع يتذكر اختيار اللغة في متصفحك أنت فقط.</p>
<h2>حفظ البيانات وحذفها</h2>
<p>نحتفظ بسجلات الاشتراك طول مدة اشتراكك وبعدها بالقدر اللي يطلبه النظام. للاطلاع على بياناتك أو تصحيحها أو حذفها راسلنا على <a href="mailto:{EMAIL}">{EMAIL}</a>.</p>
""",
)

REFUND = doc(
    "Trial, Cancellation & Refunds",
    "التجربة والإلغاء والاسترداد",
    f"""
<h2>{TRIAL_DAYS}-day free trial</h2>
<p>Every plan starts with a {TRIAL_DAYS}-day free trial so you can test StreamSafe on your own PC and setup. A payment card is required to start it. Nothing is charged during the trial.</p>
<p>If you do not cancel before the trial ends, your subscription starts automatically and the first month is charged. One free trial per person, payment card and PC.</p>
<h2>Cancel any time</h2>
<p>Cancel from the link in your purchase email (customer portal) or by emailing <a href="mailto:{EMAIL}">{EMAIL}</a>. Cancel during the trial and you pay nothing. Cancel later and you keep access until the end of the month or year you already paid for; you will not be charged again.</p>
<h2>Refunds</h2>
<p>Because the free trial lets you try StreamSafe fully before paying, payments are not refunded once charged, including partial months or years. We do refund in full when:</p>
<ul><li>you were charged twice or charged by mistake;</li><li>StreamSafe could not work on your PC at all and our support could not fix it within 7 days of the charge;</li><li>the law where you live requires a refund.</li></ul>
<p>Refunds are paid by Dodo Payments, our Merchant of Record, to the original payment method, usually within 5–10 business days. Your licence key stops working when the refund is issued.</p>
""",
    f"""
<h2>تجربة مجانية {TRIAL_DAYS} أيام</h2>
<p>كل باقة تبدأ بتجربة مجانية {TRIAL_DAYS} أيام عشان تجرب StreamSafe على جهازك وإعداداتك. يلزم إدخال بطاقة دفع لبدء التجربة، وما ينخصم شيء خلالها.</p>
<p>إذا ما ألغيت قبل نهاية التجربة يبدأ اشتراكك تلقائيًا ويُخصم أول شهر. تجربة مجانية وحدة لكل شخص وبطاقة دفع وجهاز.</p>
<h2>إلغاء في أي وقت</h2>
<p>تقدر تلغي من الرابط اللي في إيميل الشراء (بوابة العميل) أو بمراسلتنا على <a href="mailto:{EMAIL}">{EMAIL}</a>. إذا ألغيت خلال التجربة ما تدفع شيء، وإذا ألغيت بعدها يبقى اشتراكك شغّال لين نهاية الشهر أو السنة المدفوعة وما ينخصم منك مرة ثانية.</p>
<h2>الاسترداد</h2>
<p>لأن التجربة المجانية تخليك تجرب StreamSafe كامل قبل الدفع، المبالغ ما تُسترد بعد خصمها، ولا يوجد استرداد لجزء من الشهر أو السنة. ونرجع المبلغ كامل في الحالات التالية:</p>
<ul><li>إذا انخصم منك مرتين أو انخصم بالغلط؛</li><li>إذا StreamSafe ما اشتغل على جهازك أبدًا وما قدر الدعم يحل المشكلة خلال 7 أيام من الخصم؛</li><li>إذا كان النظام في بلدك يلزم بالاسترداد.</li></ul>
<p>الاسترداد يتم عن طريق Dodo Payments (البائع المسجّل) إلى نفس وسيلة الدفع، وعادةً خلال 5 إلى 10 أيام عمل. مفتاح الاشتراك يتوقف بعد الاسترداد.</p>
""",
)

CONTACT = doc(
    "Contact",
    "تواصل معنا",
    f"""
<div class="box"><p>Email: <a href="mailto:{EMAIL}">{EMAIL}</a></p></div>
<p>We reply within 2 business days (Sunday to Thursday, Saudi time). For billing questions, include the email you used to subscribe. Never send us your stream key, password or card details.</p>
<p>StreamSafe is operated by an independent developer based in Saudi Arabia.</p>
""",
    f"""
<div class="box"><p>الإيميل: <a href="mailto:{EMAIL}">{EMAIL}</a></p></div>
<p>نرد خلال يومين عمل (الأحد إلى الخميس بتوقيت السعودية). لأسئلة الدفع اكتب الإيميل اللي اشتركت فيه. لا ترسل لنا أبدًا مفتاح البث أو كلمة السر أو بيانات بطاقتك.</p>
<p>يدير StreamSafe مطوّر مستقل مقيم في المملكة العربية السعودية.</p>
""",
)

DESC = "StreamSafe covers revealing characters and scenes in your GTA 6 live stream a couple of seconds before viewers see them. Windows app for OBS, Twitch, Kick and YouTube."
PAGES = {
    "index.html": ("StreamSafe: keep your GTA 6 stream clean", HOME),
    "terms.html": ("Terms of Service | StreamSafe", TERMS),
    "privacy.html": ("Privacy Policy | StreamSafe", PRIVACY),
    "refund.html": ("Trial, Cancellation & Refunds | StreamSafe", REFUND),
    "contact.html": ("Contact | StreamSafe", CONTACT),
}

if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for f in ("style.css", "lang.js", "logo.webp", "favicon.png"):
        shutil.copy(ROOT / f, OUT / f)
    shutil.copytree(ROOT / "media", OUT / "media")
    for name, (title, body) in PAGES.items():
        (OUT / name).write_text(page(title, DESC, body), encoding="utf-8")
    print("built", sorted(p.name for p in OUT.iterdir()), "video" if HAS_VIDEO else "no video")
