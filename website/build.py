"""Builds the static StreamSafe site (one HTML per page, English + Arabic in each, toggled by lang.js)."""

import pathlib

OUT = pathlib.Path(__file__).parent / "public"
EMAIL = "streamsafe1@gmail.com"
UPDATED_EN = "17 November 2026"
UPDATED_AR = "17 نوفمبر 2026"
BASIC, PRO = 49, 99  # SAR per month


def both(en, ar, tag="div"):
    return f'<{tag} data-l="en">{en}</{tag}><{tag} data-l="ar">{ar}</{tag}>'


def page(name, title_en, body):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_en}</title>
<meta name="description" content="StreamSafe covers revealing characters in GTA 6 live streams in real time, before viewers see them. Windows app for Twitch, Kick, YouTube and more.">
<meta property="og:title" content="{title_en}">
<meta property="og:image" content="/logo.webp">
<link rel="icon" href="/favicon.png">
<link rel="stylesheet" href="/style.css">
<script src="/lang.js"></script>
</head>
<body>
<header><div class="wrap">
<a class="brand" href="/"><img src="/logo.webp" alt="StreamSafe logo">StreamSafe</a>
<nav>
{both('<a href="/#features">Features</a> <a href="/#how">How it works</a> <a href="/#pricing">Pricing</a> <a href="/contact.html">Contact</a>',
      '<a href="/#features">المزايا</a> <a href="/#how">طريقة العمل</a> <a href="/#pricing">الأسعار</a> <a href="/contact.html">تواصل معنا</a>', "span")}
</nav>
<button class="lang-btn" type="button" aria-label="Language">{both("العربية", "English", "span")}</button>
</div></header>
<main>
{body}
</main>
<footer><div class="wrap">
<span>© 2026 StreamSafe · <a href="mailto:{EMAIL}">{EMAIL}</a></span>
<span class="links">
{both(f'<a href="/#pricing">Pricing</a> <a href="/terms.html">Terms of Service</a> <a href="/privacy.html">Privacy Policy</a> <a href="/refund.html">Refund &amp; Cancellation</a> <a href="/contact.html">Contact</a>',
      f'<a href="/#pricing">الأسعار</a> <a href="/terms.html">شروط الاستخدام</a> <a href="/privacy.html">سياسة الخصوصية</a> <a href="/refund.html">الاسترداد والإلغاء</a> <a href="/contact.html">تواصل معنا</a>', "span")}
</span>
<span style="width:100%">{both("StreamSafe is an independent product. It is not affiliated with, endorsed or sponsored by Rockstar Games or Take-Two Interactive. Grand Theft Auto and GTA are their trademarks.",
      "StreamSafe منتج مستقل، وليس تابعًا لشركة Rockstar Games أو Take-Two Interactive ولا معتمدًا منهما. Grand Theft Auto و GTA علامات تجارية مملوكة لهما.", "span")}</span>
</div></footer>
</body>
</html>
"""


INDEX = f"""
<div class="hero"><div class="wrap">
<img src="/logo.webp" alt="StreamSafe">
{both('<h1>Stream GTA 6 <span class="g">without the awkward moments</span></h1><p class="lead">StreamSafe watches your gameplay a couple of seconds ahead of your viewers and covers revealing characters and scenes automatically, so your stream stays family-friendly and within platform rules.</p>',
      '<h1>بث GTA 6 <span class="g">بدون لقطات محرجة</span></h1><p class="lead">StreamSafe يشوف اللعب قبل المشاهدين بثانيتين أو ثلاث، ويغطي الشخصيات والمشاهد غير اللائقة تلقائيًا، عشان يبقى بثك نظيف ومناسب لكل العائلة ولقوانين المنصات.</p>')}
{both('<a class="btn" href="#pricing">See plans</a><a class="btn alt" href="#how">How it works</a>',
      '<a class="btn" href="#pricing">شوف الباقات</a><a class="btn alt" href="#how">طريقة العمل</a>')}
</div></div>

<section id="features"><div class="wrap">
{both('<h2>What you get</h2><p class="sub">A Windows app that sits between OBS and your streaming platform.</p>',
      '<h2>وش تحصل</h2><p class="sub">برنامج ويندوز يشتغل بين OBS ومنصة البث.</p>')}
<div class="grid">
<div class="card"><span class="ico">🛡️</span>{both("<h3>Real-time covering</h3><p>Revealing outfits are covered with a solid cloth layer; full scenes get a branded cover screen. No blur, nothing slips through.</p>",
  "<h3>تغطية فورية</h3><p>الملابس غير المحتشمة تتغطى بطبقة قماش كاملة، والمشاهد الكاملة تتغطى بشاشة بشعارنا. بدون تمويه، وما يطلع شيء للمشاهدين.</p>")}</div>
<div class="card"><span class="ico">⏱️</span>{both("<h3>Short safety delay</h3><p>A 2–3 second delay, chosen automatically for your PC, gives the app time to decide before anything reaches viewers. Game, camera and mic stay in sync.</p>",
  "<h3>تأخير بسيط للأمان</h3><p>تأخير 2 إلى 3 ثواني يختاره البرنامج حسب جهازك، عشان يقرر قبل ما يوصل أي شيء للمشاهدين. اللعبة والكاميرا والمايك يبقون متزامنين.</p>")}</div>
<div class="card"><span class="ico">📡</span>{both("<h3>All major platforms</h3><p>Twitch, Kick, YouTube, Facebook, X, Trovo and any custom RTMP/RTMPS server. Paste your address and key; we fix common mistakes.</p>",
  "<h3>كل المنصات المشهورة</h3><p>Twitch و Kick و YouTube و Facebook و X و Trovo وأي سيرفر RTMP/RTMPS. الصق الرابط والمفتاح، والبرنامج يصلح الأخطاء الشائعة.</p>")}</div>
<div class="card"><span class="ico">🎞️</span>{both("<h3>Full quality</h3><p>Your resolution and frame rate are kept (up to 1440p and 60 fps). Not live? Record a clean, covered video file for YouTube instead.</p>",
  "<h3>نفس الجودة</h3><p>الدقة وعدد الإطارات تبقى مثل ما هي (لين 1440p و 60 إطار). ما تبث مباشر؟ سجّل فيديو نظيف ومغطى وارفعه على يوتيوب.</p>")}</div>
<div class="card"><span class="ico">🎮</span>{both("<h3>PC and console</h3><p>Works with PC gameplay and with PlayStation / Xbox through a capture card into OBS.</p>",
  "<h3>بي سي وبلايستيشن</h3><p>يشتغل مع اللعب على البي سي، ومع البلايستيشن والإكس بوكس عن طريق كارت التصوير وOBS.</p>")}</div>
<div class="card"><span class="ico">🔒</span>{both("<h3>Runs on your PC</h3><p>Your video is processed locally on your own graphics card. It is never uploaded to us.</p>",
  "<h3>يشتغل على جهازك</h3><p>الفيديو يتعالج على كرت الشاشة حقك، وما ينرفع لنا أبدًا.</p>")}</div>
</div></div></section>

<section id="how"><div class="wrap">
{both('<h2>How it works</h2><p class="sub">Set up once in about five minutes.</p>', '<h2>طريقة العمل</h2><p class="sub">تجهزه مرة وحدة في حوالي خمس دقايق.</p>')}
<ol class="steps">
<li>{both("<strong>Install</strong><br>Download StreamSafe for Windows and enter your licence key.", "<strong>ثبّت</strong><br>حمّل StreamSafe على ويندوز وحط مفتاح الاشتراك.")}</li>
<li>{both("<strong>Connect</strong><br>Pick your platform and paste your stream key. The app shows the address to put in OBS.", "<strong>اربط</strong><br>اختر منصتك والصق مفتاح البث. البرنامج يعطيك العنوان اللي تحطه في OBS.")}</li>
<li>{both("<strong>Go live</strong><br>Press Start in StreamSafe, then Start Streaming in OBS.", "<strong>ابدأ البث</strong><br>اضغط تشغيل في StreamSafe، وبعدها ابدأ البث في OBS.")}</li>
<li>{both("<strong>Play</strong><br>StreamSafe covers what needs covering. If anything goes wrong, it covers the whole screen rather than risk it.", "<strong>العب</strong><br>StreamSafe يغطي اللي يحتاج تغطية. ولو صار أي خلل يغطي الشاشة كاملة بدل ما يخاطر.")}</li>
</ol>
{both('<p class="note">Requirements: Windows 10/11 (64-bit), OBS Studio, NVIDIA GeForce RTX graphics card recommended.</p>',
      '<p class="note">المتطلبات: ويندوز 10 أو 11 (64 بت)، برنامج OBS Studio، ويُنصح بكرت شاشة NVIDIA GeForce RTX.</p>')}
</div></section>

<section id="pricing"><div class="wrap">
{both('<h2>Pricing</h2><p class="sub">Monthly subscription. Cancel any time.</p>', '<h2>الأسعار</h2><p class="sub">اشتراك شهري، تقدر تلغيه في أي وقت.</p>')}
<div class="price">
<div class="plan">
{both(f'<h3>Basic</h3><p class="amount">{BASIC} SAR <small>/ month</small></p><ul><li>Real-time covering for live streams and recordings</li><li>All platforms and custom RTMP</li><li>Automatic 2–3 s delay</li><li>Full resolution and frame rate</li><li>StreamSafe cover screen</li><li>One PC</li></ul>',
      f'<h3>الأساسية</h3><p class="amount">{BASIC} ريال <small>/ شهريًا</small></p><ul><li>تغطية فورية للبث المباشر والتسجيل</li><li>كل المنصات وأي سيرفر RTMP</li><li>تأخير تلقائي 2–3 ثواني</li><li>نفس الدقة وعدد الإطارات</li><li>شاشة التغطية بشعار StreamSafe</li><li>جهاز واحد</li></ul>')}
</div>
<div class="plan pro">
{both(f'<h3>Pro</h3><p class="amount">{PRO} SAR <small>/ month</small></p><ul><li>Everything in Basic</li><li>Your own logo on the cover screen</li><li>Priority support</li><li>One PC</li></ul>',
      f'<h3>برو</h3><p class="amount">{PRO} ريال <small>/ شهريًا</small></p><ul><li>كل مزايا الباقة الأساسية</li><li>شعارك الخاص على شاشة التغطية</li><li>دعم بأولوية</li><li>جهاز واحد</li></ul>')}
</div>
</div>
{both('<p class="note">Billed monthly in Saudi riyals (shown in your local currency at checkout where available); renews automatically until cancelled. Taxes may be added at checkout depending on your country. Payments are processed by Dodo Payments, our Merchant of Record. See <a href="/refund.html">Refund &amp; Cancellation</a>.</p>',
      '<p class="note">الدفع شهري بالريال السعودي (وقد يظهر بعملتك المحلية عند الدفع)، ويتجدد تلقائيًا لين تلغيه. قد تُضاف ضرائب حسب بلدك عند الدفع. المدفوعات تتم عن طريق Dodo Payments بصفتها البائع المسجّل (Merchant of Record). اطلع على <a href="/refund.html">سياسة الاسترداد والإلغاء</a>.</p>')}
</div></section>

<section><div class="wrap">
{both(f'<h2>Questions?</h2><p class="sub">Email us at <a href="mailto:{EMAIL}">{EMAIL}</a>. We usually reply within 2 business days.</p>',
      f'<h2>عندك سؤال؟</h2><p class="sub">راسلنا على <a href="mailto:{EMAIL}">{EMAIL}</a>، وعادةً نرد خلال يومين عمل.</p>')}
</div></section>
"""


def doc(title_en, title_ar, en, ar):
    return f"""<div class="doc">
<div data-l="en"><h1>{title_en}</h1><p class="updated">Last updated: {UPDATED_EN}</p>{en}</div>
<div data-l="ar" dir="rtl"><h1>{title_ar}</h1><p class="updated">آخر تحديث: {UPDATED_AR}</p>{ar}</div>
</div>"""


TERMS = doc("Terms of Service", "شروط الاستخدام", f"""
<p>These terms apply to the StreamSafe Windows application, its website and its subscriptions ("StreamSafe", "we", "us"). StreamSafe is operated by an independent developer based in Saudi Arabia. By buying a subscription or using the app you agree to these terms.</p>
<h2>1. The service</h2>
<p>StreamSafe is software that runs on your own PC. It delays your stream by a few seconds and automatically covers characters and scenes it judges to be revealing in Grand Theft Auto VI gameplay before the video is sent to your streaming platform or saved to a file.</p>
<h2>2. No guarantee of perfect detection</h2>
<p>StreamSafe is designed to cover more rather than less, and to cover the whole screen when it is unsure or when something fails. Even so, automatic detection cannot be perfect. You remain responsible for the content of your streams and recordings and for following the rules of the platforms you use. We are not liable for platform actions such as warnings, suspensions or bans.</p>
<h2>3. Subscriptions and payment</h2>
<p>Subscriptions are billed monthly in advance at the price shown on our <a href="/#pricing">pricing section</a> and renew automatically until cancelled. Our order process is conducted by our online reseller Dodo Payments. Dodo Payments is the Merchant of Record for all our orders and handles payment, billing, taxes and invoices. Prices may change; any change applies from your next billing period and we will tell you in advance by email.</p>
<h2>4. Licence</h2>
<p>Each subscription gives you a personal, non-transferable licence key for one PC at a time. You may move it to another PC by deactivating it in the app. You may not share, resell, copy, reverse engineer or redistribute the app or its models, or try to bypass the licence check.</p>
<h2>5. Cancellation and refunds</h2>
<p>You can cancel at any time. Refunds are described in our <a href="/refund.html">Refund &amp; Cancellation Policy</a>.</p>
<h2>6. Acceptable use</h2>
<p>Use StreamSafe only for lawful purposes and only with streaming accounts and content you are entitled to use. Do not use it to break any law or any platform's terms.</p>
<h2>7. Updates and availability</h2>
<p>We may update the app to improve detection, fix problems or support new platforms. The app needs an internet connection at least every few days to confirm your subscription.</p>
<h2>8. Liability</h2>
<p>To the extent permitted by law, StreamSafe is provided "as is", and our total liability for any claim is limited to the amount you paid us in the three months before the claim.</p>
<h2>9. Trademarks</h2>
<p>StreamSafe is not affiliated with, endorsed or sponsored by Rockstar Games or Take-Two Interactive. Grand Theft Auto and GTA are their trademarks.</p>
<h2>10. Changes and contact</h2>
<p>We may update these terms and will post the new version here. Questions: <a href="mailto:{EMAIL}">{EMAIL}</a>.</p>
""", f"""
<p>هذه الشروط تنطبق على برنامج StreamSafe لويندوز وموقعه واشتراكاته ("StreamSafe" أو "نحن"). يدير StreamSafe مطوّر مستقل مقيم في المملكة العربية السعودية. بشرائك اشتراكًا أو استخدامك البرنامج فأنت توافق على هذه الشروط.</p>
<h2>1. الخدمة</h2>
<p>StreamSafe برنامج يشتغل على جهازك. يؤخر البث بضع ثوانٍ ويغطي تلقائيًا الشخصيات والمشاهد غير اللائقة في لعبة Grand Theft Auto VI قبل إرسال الفيديو لمنصة البث أو حفظه في ملف.</p>
<h2>2. لا يوجد ضمان لكشف كامل</h2>
<p>StreamSafe مصمم ليغطي أكثر لا أقل، ويغطي الشاشة كاملة إذا ما كان متأكد أو إذا صار خلل. ومع ذلك فالكشف التلقائي لا يمكن أن يكون كاملًا. تبقى أنت المسؤول عن محتوى بثك وتسجيلاتك وعن الالتزام بقوانين المنصات، ولسنا مسؤولين عن إجراءات المنصات مثل التحذير أو الإيقاف أو الحظر.</p>
<h2>3. الاشتراكات والدفع</h2>
<p>الاشتراك يُدفع شهريًا مقدمًا بالسعر الظاهر في <a href="/#pricing">قسم الأسعار</a> ويتجدد تلقائيًا لين تلغيه. تتم عملية الشراء عن طريق بائعنا الإلكتروني Dodo Payments، وهو البائع المسجّل (Merchant of Record) لجميع الطلبات ويتولى الدفع والفوترة والضرائب والفواتير. قد تتغير الأسعار، وأي تغيير يطبق من فترة الفوترة التالية ونبلغك مسبقًا بالإيميل.</p>
<h2>4. الترخيص</h2>
<p>كل اشتراك يعطيك مفتاح ترخيص شخصي غير قابل للتحويل لجهاز واحد في نفس الوقت، وتقدر تنقله لجهاز ثاني بإلغاء تفعيله من البرنامج. لا يجوز مشاركة البرنامج أو نماذجه أو بيعها أو نسخها أو الهندسة العكسية لها أو تجاوز فحص الاشتراك.</p>
<h2>5. الإلغاء والاسترداد</h2>
<p>تقدر تلغي في أي وقت، والاسترداد موضح في <a href="/refund.html">سياسة الاسترداد والإلغاء</a>.</p>
<h2>6. الاستخدام المقبول</h2>
<p>استخدم StreamSafe لأغراض مشروعة فقط، ومع حسابات بث ومحتوى يحق لك استخدامها، ولا تستخدمه لمخالفة أي نظام أو شروط أي منصة.</p>
<h2>7. التحديثات والتوفر</h2>
<p>قد نحدّث البرنامج لتحسين الكشف أو إصلاح المشاكل أو دعم منصات جديدة. يحتاج البرنامج اتصال إنترنت مرة على الأقل كل بضعة أيام للتأكد من الاشتراك.</p>
<h2>8. المسؤولية</h2>
<p>في حدود ما يسمح به النظام، يُقدَّم StreamSafe "كما هو"، ومسؤوليتنا الكاملة عن أي مطالبة محدودة بالمبلغ الذي دفعته لنا خلال الأشهر الثلاثة السابقة للمطالبة.</p>
<h2>9. العلامات التجارية</h2>
<p>StreamSafe ليس تابعًا لشركة Rockstar Games أو Take-Two Interactive ولا معتمدًا منهما. Grand Theft Auto و GTA علامات تجارية مملوكة لهما.</p>
<h2>10. التعديلات والتواصل</h2>
<p>قد نعدّل هذه الشروط وننشر النسخة الجديدة هنا. للأسئلة: <a href="mailto:{EMAIL}">{EMAIL}</a>.</p>
""")

PRIVACY = doc("Privacy Policy", "سياسة الخصوصية", f"""
<p>This policy explains what information StreamSafe handles and why.</p>
<h2>Your video and audio stay on your PC</h2>
<p>StreamSafe processes your gameplay, camera and microphone locally on your own computer. We never receive, upload, store or watch your video or audio. It goes only to the streaming platform or file you choose.</p>
<h2>What we do process</h2>
<ul>
<li><strong>Licence check:</strong> when you activate the app and every few hours while it runs, the app sends your licence key, an activation ID and your PC's name to Dodo Payments to confirm your subscription is active.</li>
<li><strong>Purchase details:</strong> when you subscribe, Dodo Payments (our Merchant of Record) collects your name, email, billing country and payment details. We receive your email, plan and subscription status. We never see your full card details. See Dodo Payments' privacy policy at <a href="https://dodopayments.com/privacy-policy">dodopayments.com</a>.</li>
<li><strong>Updates:</strong> the app may download update and model files from our servers; this shares your IP address with the hosting provider, as any download does.</li>
<li><strong>Support emails:</strong> if you email us, we keep the conversation to help you.</li>
</ul>
<h2>Stream keys</h2>
<p>Your stream key is stored only on your PC, in the app's settings, and is only sent to the platform you stream to.</p>
<h2>What we don't do</h2>
<p>We don't sell your data, show ads, or use tracking cookies on this website.</p>
<h2>Retention and your rights</h2>
<p>We keep subscription records for as long as your subscription is active and as long as the law requires afterwards. You can ask us to access, correct or delete your information at <a href="mailto:{EMAIL}">{EMAIL}</a>.</p>
""", f"""
<p>هذه السياسة توضح وش المعلومات اللي يتعامل معها StreamSafe وليش.</p>
<h2>الفيديو والصوت يبقون على جهازك</h2>
<p>StreamSafe يعالج اللعب والكاميرا والمايك على جهازك فقط. ما نستلم الفيديو أو الصوت ولا نرفعه ولا نحفظه ولا نشوفه أبدًا، ويروح فقط للمنصة أو الملف اللي تختاره.</p>
<h2>وش اللي نعالجه</h2>
<ul>
<li><strong>فحص الاشتراك:</strong> عند تفعيل البرنامج وكل بضع ساعات أثناء تشغيله، يرسل البرنامج مفتاح الاشتراك ورقم التفعيل واسم جهازك إلى Dodo Payments للتأكد أن اشتراكك فعّال.</li>
<li><strong>بيانات الشراء:</strong> عند الاشتراك تجمع Dodo Payments (البائع المسجّل) اسمك وإيميلك وبلدك وبيانات الدفع. نحن نستلم الإيميل والباقة وحالة الاشتراك فقط، وما نشوف بيانات بطاقتك كاملة. اطلع على سياسة خصوصية Dodo Payments في <a href="https://dodopayments.com/privacy-policy">dodopayments.com</a>.</li>
<li><strong>التحديثات:</strong> قد يحمّل البرنامج ملفات التحديث والنماذج من سيرفراتنا، وهذا يشارك عنوان IP مع مزود الاستضافة مثل أي تحميل.</li>
<li><strong>رسائل الدعم:</strong> إذا راسلتنا نحتفظ بالمحادثة عشان نساعدك.</li>
</ul>
<h2>مفاتيح البث</h2>
<p>مفتاح البث يُحفظ على جهازك فقط في إعدادات البرنامج، ولا يُرسل إلا للمنصة اللي تبث عليها.</p>
<h2>وش اللي ما نسويه</h2>
<p>ما نبيع بياناتك، وما نعرض إعلانات، وما نستخدم كوكيز تتبع في هذا الموقع.</p>
<h2>مدة الحفظ وحقوقك</h2>
<p>نحتفظ بسجلات الاشتراك طول مدة اشتراكك وبعدها بالقدر اللي يطلبه النظام. تقدر تطلب الاطلاع على بياناتك أو تصحيحها أو حذفها على <a href="mailto:{EMAIL}">{EMAIL}</a>.</p>
""")

REFUND = doc("Refund & Cancellation Policy", "سياسة الاسترداد والإلغاء", f"""
<h2>Cancel any time</h2>
<p>You can cancel your subscription at any time from the link in your purchase email (customer portal) or by emailing <a href="mailto:{EMAIL}">{EMAIL}</a>. After cancelling you keep access until the end of the month you already paid for, and you will not be charged again.</p>
<h2>7-day refund on your first payment</h2>
<p>If StreamSafe does not work for you, email us within 7 days of your first payment and we will refund it in full. No complicated questions; a short description of the problem helps us improve.</p>
<h2>Renewals</h2>
<p>Monthly renewals are not refundable once the new month has started, except if you forgot to cancel and email us within 48 hours of the renewal without having used the app since, or where the law requires a refund.</p>
<h2>How refunds are paid</h2>
<p>Refunds are issued by Dodo Payments, our Merchant of Record, to the original payment method. They usually appear within 5–10 business days depending on your bank. Your licence key stops working when the refund is issued.</p>
""", f"""
<h2>إلغاء في أي وقت</h2>
<p>تقدر تلغي اشتراكك في أي وقت من الرابط الموجود في إيميل الشراء (بوابة العميل) أو بمراسلتنا على <a href="mailto:{EMAIL}">{EMAIL}</a>. بعد الإلغاء يبقى اشتراكك شغّال لين نهاية الشهر اللي دفعته، وما ينخصم منك مرة ثانية.</p>
<h2>استرداد خلال 7 أيام لأول دفعة</h2>
<p>إذا ما اشتغل StreamSafe معك، راسلنا خلال 7 أيام من أول دفعة ونرجع لك المبلغ كامل. بدون أسئلة معقدة، ووصف بسيط للمشكلة يساعدنا نتحسن.</p>
<h2>التجديد</h2>
<p>التجديد الشهري ما يُسترد بعد بداية الشهر الجديد، إلا إذا نسيت تلغي وراسلتنا خلال 48 ساعة من التجديد وما استخدمت البرنامج بعده، أو إذا كان النظام يلزم بالاسترداد.</p>
<h2>طريقة الاسترداد</h2>
<p>الاسترداد يتم عن طريق Dodo Payments (البائع المسجّل) إلى نفس وسيلة الدفع، وعادةً يظهر خلال 5 إلى 10 أيام عمل حسب البنك. مفتاح الاشتراك يتوقف بعد الاسترداد.</p>
""")

CONTACT = doc("Contact", "تواصل معنا", f"""
<p>Email: <a href="mailto:{EMAIL}">{EMAIL}</a></p>
<p>We reply within 2 business days (Sunday–Thursday, Saudi time). For billing questions, please include the email you used to subscribe. Please never send us your stream key or card details.</p>
<p>StreamSafe is operated by an independent developer based in Saudi Arabia.</p>
""", f"""
<p>الإيميل: <a href="mailto:{EMAIL}">{EMAIL}</a></p>
<p>نرد خلال يومين عمل (الأحد إلى الخميس بتوقيت السعودية). لأسئلة الدفع اكتب الإيميل اللي اشتركت فيه. لا ترسل لنا أبدًا مفتاح البث أو بيانات بطاقتك.</p>
<p>يدير StreamSafe مطوّر مستقل مقيم في المملكة العربية السعودية.</p>
""")

PAGES = {
    "index.html": ("StreamSafe – keep your GTA 6 stream clean", INDEX),
    "terms.html": ("Terms of Service – StreamSafe", TERMS),
    "privacy.html": ("Privacy Policy – StreamSafe", PRIVACY),
    "refund.html": ("Refund & Cancellation Policy – StreamSafe", REFUND),
    "contact.html": ("Contact – StreamSafe", CONTACT),
}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    root = pathlib.Path(__file__).parent
    for f in ("style.css", "lang.js", "logo.webp", "favicon.png"):
        (OUT / f).write_bytes((root / f).read_bytes())
    for name, (title, body) in PAGES.items():
        (OUT / name).write_text(page(name, title, body), encoding="utf-8")
    print("built", sorted(p.name for p in OUT.iterdir()))
