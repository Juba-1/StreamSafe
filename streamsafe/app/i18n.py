"""Every word the window shows, in Arabic and English. Arabic is the default.

``t(lang, key, **values)`` returns the text. A key missing in one language falls back to the other, then to the key
itself: the window never fails because of a missing string (a test checks that no key is missing)."""

from __future__ import annotations

from streamsafe.branding import PRODUCT_NAME

LANGS = ("ar", "en")

# key: (Arabic, English). {name} is the product name, other {fields} are filled by the caller.
TEXT: dict[str, tuple[str, str]] = {
    "title": ("{name}", "{name}"),
    "lang_button": ("English", "عربي"),
    "tab_home": ("الرئيسية", "Home"),
    "tab_settings": ("إعدادات متقدمة", "Advanced"),
    "tab_check": ("الفحص", "Check"),
    # --- status badge ---
    "st_stopped": ("متوقف", "Stopped"),
    "st_starting": ("جاري التشغيل... الشاشة مغطاة", "Starting... screen is covered"),
    "st_protected": ("محمي", "Protected"),
    "st_covered": ("مغطى بالكامل", "Fully covered"),
    "st_stalled": ("المحرك لا يرد: الشاشة مغطاة", "Engine not answering: screen covered"),
    "st_failed": ("توقف بسبب خطأ", "Stopped by an error"),
    "st_blocked": ("غير جاهز", "Not ready"),
    "st_hint_protected": ("البث يمر عبر الحماية.", "The stream is going through the protection."),
    "st_hint_covered": (
        "الشاشة مغطاة كاملة الآن، لن يرى المشاهدون شيئًا حتى يتأكد البرنامج.",
        "The whole picture is covered; viewers see nothing until the program is sure.",
    ),
    "st_hint_stopped": ("اضغط «ابدأ الحماية» عشان يبدأ البث المحمي.", "Press Start protection to begin the protected stream."),
    # --- beta ---
    "beta_days": ("نسخة تجريبية: تنتهي بعد {days} يوم ({date}).", "Beta version: ends in {days} days ({date})."),
    "beta_last_day": ("نسخة تجريبية: تنتهي اليوم.", "Beta version: ends today."),
    "beta_expired": (
        "انتهت هذه النسخة التجريبية ولن تبدأ الحماية. نزّل النسخة الجديدة من الفريق ثم ثبّتها.",
        "This beta version has ended and will not start protection. Download the new version from the team and install it.",
    ),
    # --- system check ---
    "sec_check": ("فحص الجهاز", "System check"),
    "recheck": ("افحص من جديد", "Check again"),
    "checking": ("جاري الفحص...", "Checking..."),
    "check_failed": ("تعذر الفحص: {why}", "The check could not run: {why}"),
    "verdict_ok": ("جهازك جاهز.", "Your PC is ready."),
    "verdict_warn": ("جهازك يشتغل لكن فيه ملاحظات.", "Your PC works, with remarks."),
    "verdict_fail": ("جهازك غير جاهز: صلّح الأحمر أولًا.", "Your PC is not ready: fix the red items first."),
    "chk_os": ("النظام", "Windows"),
    "chk_python": ("بيئة التشغيل", "Runtime"),
    "chk_gpu": ("كرت الشاشة", "Graphics card"),
    "chk_runtime": ("تشغيل الكرت", "GPU runtime"),
    "chk_ram": ("الذاكرة", "Memory"),
    "chk_disk": ("المساحة", "Disk space"),
    "chk_ffmpeg": ("FFmpeg", "FFmpeg"),
    "chk_obs": ("برنامج OBS", "OBS"),
    "chk_models": ("ملفات النماذج", "Model files"),
    "chk_engines": ("تجهيز كرت إنفيديا", "NVIDIA preparation"),
    "chk_heads": ("ملفات القرار", "Decision files"),
    # --- setup (models / NVIDIA) ---
    "sec_setup": ("التجهيز", "Setup"),
    "models_needed": (
        "النماذج غير مثبتة. تنزيلها يحتاج اتصال إنترنت ومساحة كبيرة (مرة واحدة).",
        "The models are not installed. Downloading needs internet and a lot of disk space (once).",
    ),
    "models_ok": ("النماذج جاهزة.", "Models are ready."),
    "models_bad": ("النماذج ناقصة أو تالفة: {why}", "Models are missing or damaged: {why}"),
    "download_models": ("نزّل النماذج", "Download models"),
    "choose_folder": ("اختر مجلد النماذج...", "Choose models folder..."),
    "choose_zip": ("اختر ملف zip...", "Choose zip file..."),
    "pick_folder_title": ("اختر المجلد اللي فيه ملفات النماذج", "Choose the folder with the model files"),
    "pick_zip_title": ("اختر ملف النماذج (zip)", "Choose the models file (zip)"),
    "cancel": ("إلغاء", "Cancel"),
    "src_private": (
        "ما قدرنا نفتح مكان النماذج: غير متاح للعامة بعد أو يحتاج تسجيل دخول ({why}). اختر ملف النماذج اللي أرسلناه لك (zip أو مجلد).",
        "We could not open the model download: it is not public yet or needs a login ({why}). Choose the models file we sent you (zip or folder).",
    ),
    "src_missing": (
        "ما لقينا النماذج في مكان التنزيل ({why}). اختر ملف النماذج اللي أرسلناه لك (zip أو مجلد).",
        "The models were not found at the download address ({why}). Choose the models file we sent you (zip or folder).",
    ),
    "src_network": (
        "ما قدرنا نتصل بالإنترنت لتنزيل النماذج ({why}). تأكد من الاتصال وحاول من جديد، أو اختر الملف اللي أرسلناه لك.",
        "No connection for the model download ({why}). Check your internet and try again, or choose the file we sent you.",
    ),
    "src_other": ("تعذر تنزيل النماذج: {why}", "The models could not be downloaded: {why}"),
    "pack_bad": ("الملف ليس حزمة نماذج سليمة: {why}", "That is not a good model pack: {why}"),
    "installing": ("جاري التنزيل / التثبيت...", "Downloading / installing..."),
    "p_downloading": ("جاري التنزيل", "Downloading"),
    "p_unpacking": ("فك الضغط", "Unpacking"),
    "p_copying": ("نسخ الملفات", "Copying files"),
    "p_verifying": ("التحقق من الملفات", "Verifying files"),
    "install_done": ("تم تثبيت النماذج.", "The models are installed."),
    "install_cancelled": ("أُلغي التثبيت. ما تغير شي.", "Install cancelled. Nothing was changed."),
    "nvidia_offer": (
        "عندك كرت إنفيديا: الوضع السريع يحتاج تنزيل مكتبات إنفيديا (حوالي {gb} جيجا) ثم تجهيز الكرت (حوالي 4 دقائق، مرة واحدة). بدونه البرنامج يشتغل لكن أبطأ.",
        "You have an NVIDIA card: the fast mode downloads NVIDIA libraries (about {gb} GB) and then prepares the card (about 4 minutes, once). Without it the program works, only slower.",
    ),
    "nvidia_button": ("فعّل الوضع السريع لإنفيديا", "Enable fast NVIDIA mode"),
    "nvidia_later": ("لا، لاحقًا", "Not now"),
    "nvidia_ready": ("الوضع السريع لإنفيديا جاهز.", "Fast NVIDIA mode is ready."),
    "nvidia_downloading": ("تنزيل مكتبات إنفيديا", "Downloading NVIDIA libraries"),
    "nvidia_preparing": ("تجهيز كرت إنفيديا (حوالي 4 دقائق)", "Preparing the NVIDIA card (about 4 minutes)"),
    "nvidia_failed": (
        "تعذر تفعيل الوضع السريع: {why}. البرنامج يكمل بالوضع العادي.",
        "Fast mode could not be enabled: {why}. The program continues in the normal mode.",
    ),
    # --- settings ---
    "sec_in": ("مصدر الصورة", "Where the picture comes from"),
    "sec_in_note": ("الأفضل تخليه على OBS. غيّره بس إذا تعرف وش تسوي.", "Leave it on OBS unless you know you need something else."),
    "in_srt": (
        "من OBS (ألعاب الكمبيوتر، أو البلايستيشن والإكس بوكس عن طريق كرت التقاط داخل OBS) - المستحسن",
        "From OBS (PC games, or PlayStation / Xbox through a capture card inside OBS) - recommended",
    ),
    "in_camera": ("كرت التقاط أو كاميرا مباشرة (بدون OBS)", "A capture card or camera directly (without OBS)"),
    "srt_port": ("المنفذ", "Port"),
    "srt_hint": ("سيرفر OBS: {server}", "OBS server: {server}"),
    "camera_name": ("الجهاز", "Device"),
    "sec_out": ("وين يروح الفيديو المحمي", "Where the protected video goes"),
    "sec_out_note": (
        "البث المباشر يستخدم المنصة والمفتاح اللي في الصفحة الرئيسية.",
        "Live streaming uses the platform and key on the Home tab.",
    ),
    "out_window": ("نافذة يلتقطها OBS (متقدم)", "A window that OBS captures (advanced)"),
    "out_rtmp": (
        "بث مباشر على منصة (كيك، تيك توك، يوتيوب، تويتش، فيسبوك...)",
        "Live on a platform (Kick, TikTok, YouTube, Twitch, Facebook...)",
    ),
    "rtmp_url": ("عنوان السيرفر", "Server address"),
    "stream_key": ("مفتاح البث", "Stream key"),
    "remember_key": ("احفظ المفتاح على هذا الجهاز (محمي بحساب ويندوز)", "Remember the key on this PC (protected by your Windows account)"),
    "key_no_store": (
        "هذا الجهاز ما فيه مخزن محمي: المفتاح يبقى فقط طول ما النافذة مفتوحة.",
        "This PC has no protected store: the key stays only while the window is open.",
    ),
    "delay": ("التأخير", "Delay"),
    "translate_ar": ("ترجمة النصوص للعربي (تجريبي)", "Translate on-screen text to Arabic (experimental)"),
    "translate_hint": (
        "تُرسم الترجمة على البث فقط (لا تظهر في لعبتك). أول تفعيل ينزّل ملفات (حوالي {mb} ميجا). لو تعذّر شي يكمل البث بدونها.",
        "The translation is drawn on the stream only (not in your own game). Switching it on the first time downloads files (about {mb} MB). If anything fails, the stream carries on without it.",
    ),
    "translate_downloading": ("تنزيل ملفات الترجمة", "Downloading translation files"),
    "translate_ready": ("ملفات الترجمة جاهزة.", "The translation files are ready."),
    "translate_failed": (
        "تعذّر تجهيز الترجمة ({why}). البث يكمل بدونها.",
        "Translation could not be set up ({why}). The stream carries on without it.",
    ),
    "bad_port": ("المنفذ لازم يكون بين 1024 و 65535.", "The port must be between 1024 and 65535."),
    "bad_camera": ("اكتب اسم الكاميرا.", "Enter the camera name."),
    "bad_rtmp": (
        "اختر منصتك في الخطوة 1، أو الصق عنوان السيرفر (يبدأ بـ rtmp:// أو rtmps://).",
        "Pick your platform in step 1, or paste the server address (it starts with rtmp:// or rtmps://).",
    ),
    "bad_delay": ("التأخير لازم يكون بين 0.5 و 10 ثواني.", "The delay must be between 0.5 and 10 seconds."),
    "bad_key": ("اكتب مفتاح البث (بدون مسافات).", "Enter the stream key (no spaces)."),
    # --- run ---
    "pass_on": (
        "برو مفعّل بمفتاح الداعمين لين {date} (باسم: {who}). شكراً لدعمك ♥",
        "Pro is on with a supporter key until {date} (for: {who}). Thank you for your support ♥",
    ),
    "pass_ended": (
        "مفتاح الداعمين انتهى في {date}. رجع البرنامج للخطة العادية، والحماية شغالة كاملة.",
        "The supporter key ended on {date}. The app is back on the normal plan; protection works fully.",
    ),
    "pass_invalid": ("المفتاح هذا غير صحيح. انسخه كامل من الرسالة بدون نقص.", "This key isn't valid. Copy it whole from the message."),
    "pass_expired": ("المفتاح هذا انتهت مدته.", "This key has already ended."),
    "pass_clock": (
        "ساعة الجهاز ورا التاريخ الصحيح، فوقّفنا Pro. صحّح التاريخ والوقت في ويندوز وافتح البرنامج من جديد.",
        "The PC clock is behind the real date, so Pro is off. Fix the date and time in Windows and open the app again.",
    ),
    "support_title": ("♥  عاجبك StreamSafe؟", "♥  Enjoying StreamSafe?"),
    "support_text": (
        "البرنامج مجاني وقت إطلاق GTA 6. لو حاب تساعدنا نكمل ونطوّره، دعمك ولو بقهوة يفرق معنا كثير.",
        "It's free for the GTA 6 launch. If you'd like to help us keep improving it, even a coffee means a lot to us.",
    ),
    "support_button": ("ادعمنا على Ko-fi", "Support us on Ko-fi"),
    "support_sub": ("يفتح صفحة الدعم في المتصفح", "Opens our support page"),
    "start_sub": ("يشغّل التغطية ويبدأ البث", "Covers and starts your stream"),
    "stop_sub": ("يوقف البث بأمان", "Ends the stream safely"),
    "quick_title": ("جهّز بثك في ٣ خطوات", "Set up your stream in 3 steps"),
    "quick_note": ("مرة وحدة بس، والبرنامج يتذكرها.", "Only once; StreamSafe remembers it."),
    "step_platform": ("اختر منصتك", "Pick your platform"),
    "step_key": ("الصق مفتاح البث", "Paste your stream key"),
    "step_obs": ("OBS: نجهّزه لك", "OBS: we set it up for you"),
    "step_obs_note": (
        "سكّر OBS واضغط الزر مرة وحدة. بعدها كل ما تضغط «ابدأ الحماية» نفتح OBS ويبدأ يرسل لنا لحاله.",
        "Close OBS and press the button once. After that, every «Start protection» opens OBS and it starts sending to us by itself.",
    ),
    "obs_auto_btn": ("اضبط OBS تلقائياً", "Set up OBS for me"),
    "obs_auto_done": ("تم ✓ OBS جاهز، ما تحتاج تغيّر فيه شي.", "Done ✓ OBS is ready; nothing to change in it."),
    "obs_auto_running": (
        "OBS مفتوح: سكّره أول (نعدّل إعداداته وهو مقفل)، وبعدها اضغط «اضبط OBS تلقائياً» أو «ابدأ الحماية» مرة ثانية.",
        "OBS is open: close it first (we change its settings while it is closed), then press «Set up OBS for me» or «Start protection» again.",
    ),
    "obs_auto_no_obs": (
        "ما لقينا OBS على جهازك. نزّله من obsproject.com وافتحه مرة وحدة وسكّره، وبعدها ارجع هنا.",
        "OBS isn't on this PC. Get it from obsproject.com, open it once and close it, then come back here.",
    ),
    "obs_auto_error": (
        "ما قدرنا نعدّل إعدادات OBS. استخدم العنوان تحت وطريقة الربط خطوة بخطوة.",
        "We couldn't change OBS's settings. Use the address below and the step-by-step guide.",
    ),
    "obs_opening": (
        "نفتح OBS ويبدأ يرسل لـ StreamSafe لحاله. لو ما بدأ خلال ثواني، اضغط Start Streaming في OBS.",
        "Opening OBS; it starts sending to StreamSafe by itself. If it hasn't within a few seconds, press Start Streaming in OBS.",
    ),
    "obs_press_start": ("OBS مفتوح وجاهز: اضغط Start Streaming فيه.", "OBS is open and ready: press Start Streaming in it."),
    "obs_manual": (
        "أو بنفسك: في OBS الخدمة «مخصص» والسيرفر هذا العنوان، والمفتاح فاضي.",
        "Or by hand: in OBS, Service «Custom», this address as Server, key empty.",
    ),
    "console_hint": (
        "تلعب على بلايستيشن أو إكس بوكس؟ طفّ HDCP وHDR من إعدادات الجهاز، ودخّل كارت التصوير في OBS مثل العادة.",
        "Playing on PlayStation or Xbox? Turn off HDCP and HDR in the console settings, then add your capture card in OBS as usual.",
    ),
    "sec_translate": ("ترجمة صوت اللاعبين", "Translate voice chat"),
    "start": ("ابدأ الحماية", "Start protection"),
    "stop": ("أوقف", "Stop"),
    "stopping": ("جاري الإيقاف وكتابة التقرير...", "Stopping and writing the report..."),
    "open_report": ("افتح آخر تقرير جلسة", "Open last session report"),
    "no_report": ("ما فيه تقرير بعد.", "There is no report yet."),
    "open_obs": ("إعداد OBS", "OBS setup"),
    "obs_state": ("ملف الحالة لسكربت OBS: {path}", "State file for the OBS script: {path}"),
    "obs_script": ("سكربت الحماية: {path}", "Guard script: {path}"),
    "copy": ("انسخ", "Copy"),
    "sec_brand": ("شعارك على شاشة التغطية", "Your logo on the full-screen cover"),
    "brand_hint": (
        "لك تحط شعارك أو شعار الراعي على الغطاء بدل شعار StreamSafe: على لبس الشخصيات وعلى الشاشة الكاملة.",
        "You may put your logo or your sponsor's on the covers instead of StreamSafe's: on the characters' clothing and on the full-screen card.",
    ),
    "brand_pick": ("اختر صورة...", "Choose an image..."),
    "brand_remove": ("إزالة", "Remove"),
    "brand_none": ("ما فيه شعار (الغطاء العادي)", "No logo (the plain cover)"),
    "brand_set": ("الشعار: {name}", "Logo: {name}"),
    "brand_bad": ("ما قدرنا نقرأ الصورة. اختر PNG أو JPG.", "Could not read that image. Choose a PNG or JPG."),
    "brand_pick_title": ("اختر شعارك", "Choose your logo"),
    "not_ready": ("ما يمديك تبدأ الآن: {why}", "Cannot start now: {why}"),
    "why_models": ("النماذج غير جاهزة", "the models are not ready"),
    "why_heads": ("ملفات القرار غير موجودة أو غير صالحة", "the decision files are missing or invalid"),
    "why_heads_placeholder": (
        "هذه نسخة اختبار بملفات قرار تجريبية: لا تبث بها",
        "this is a test build with placeholder decision files: do not stream with it",
    ),
    "why_expired": ("انتهت النسخة التجريبية", "the beta ended"),
    "why_check": ("الفحص فيه أخطاء حمراء", "the check has red items"),
    "why_running": ("الحماية تعمل أصلًا", "protection is already running"),
    "engine_exit": ("توقف المحرك (الرمز {code}). الشاشة صارت مغطاة.", "The engine stopped (code {code}). The screen is covered."),
    "engine_expired": ("انتهت النسخة التجريبية: لم يبدأ البث.", "The beta ended: nothing was streamed."),
    "engine_no_encoder": ("ما لقينا مشفّر فيديو يشتغل على جهازك: لم يبدأ البث.", "No video encoder works on this PC: nothing was streamed."),
    "engine_no_heads": ("ملفات القرار غير صالحة: لم يبدأ البث.", "The decision files are not valid: nothing was streamed."),
    "camera_sound": ("صوت الجهاز (لكرت الالتقاط)", "Device sound (for a capture card)"),
    "refresh_devices": ("دوّر على الأجهزة", "Find devices"),
    "no_devices": (
        "ما لقينا كرت التقاط ولا كاميرا. تأكد إنه موصول وإن برنامج ثاني ما ماسكه، ثم جرّب مرة ثانية.",
        "No capture card or camera was found. Check that it is plugged in and not held by another program, then try again.",
    ),
    "camera_hint": (
        "لو تستخدم OBS خلّك على الخيار الأول: OBS يجمع اللعبة والكاميرا والمايك وبرنامجنا يحميهم كلهم.",
        "If you use OBS, keep the first choice: OBS mixes the game, camera and mic, and we protect all of it.",
    ),
    "in_file": ("فيديو مسجّل من قبل (أحميه وأحفظه كفيديو جديد)", "A video already recorded (protect it and save a new video)"),
    "input_file": ("الملف", "File"),
    "pick_file": ("اختر فيديو...", "Choose a video..."),
    "videos": ("فيديوهات", "Videos"),
    "platform": ("المنصة", "Platform"),
    "plat_kick": ("كيك Kick", "Kick"),
    "plat_twitch": ("تويتش Twitch", "Twitch"),
    "plat_youtube": ("يوتيوب YouTube", "YouTube"),
    "plat_tiktok": ("تيك توك TikTok", "TikTok"),
    "plat_facebook": ("فيسبوك Facebook", "Facebook"),
    "plat_custom": ("منصة ثانية (اكتب العنوان)", "Another platform (type the address)"),
    "plat_hint_kick": (
        "من كيك: لوحة التحكم > الإعدادات > Stream URL & Key. الصق المفتاح بس (والعنوان لو يختلف).",
        "On Kick: Dashboard > Settings > Stream URL & Key. Paste the key (and the URL if yours differs).",
    ),
    "plat_hint_twitch": (
        "من تويتش: Creator Dashboard > Settings > Stream > Primary Stream Key. الصق المفتاح بس.",
        "On Twitch: Creator Dashboard > Settings > Stream > Primary Stream Key. Paste the key only.",
    ),
    "plat_hint_youtube": (
        "من يوتيوب: YouTube Studio > Go live > Stream > Stream key. الصق المفتاح بس.",
        "On YouTube: YouTube Studio > Go live > Stream > Stream key. Paste the key only.",
    ),
    "plat_hint_tiktok": (
        "تيك توك: لو عندك LIVE Producer (عادة 1000 متابع وفوق) انسخ «Server URL» و«Stream Key» من livecenter.tiktok.com، ويتغيرون كل بث. ما عندك؟ اختر «نافذة يلتقطها OBS» وخلّ TikTok LIVE Studio يلتقط نافذة «StreamSafe Output».",
        "TikTok: with LIVE Producer (usually 1000+ followers) copy the Server URL and Stream Key from livecenter.tiktok.com; they change every stream. Without it: choose «A window that OBS captures» and let TikTok LIVE Studio capture the «StreamSafe Output» window.",
    ),
    "plat_hint_facebook": (
        "من فيسبوك: Live Producer > Streaming software > Stream key. الصق المفتاح بس.",
        "On Facebook: Live Producer > Streaming software > Stream key. Paste the key only.",
    ),
    "plat_hint_custom": (
        "الصق عنوان السيرفر والمفتاح من المنصة. لو لصقت العنوان كامل مع المفتاح نفصلهم لك.",
        "Paste the server address and the key from the platform. If you paste the full address with the key, we split it for you.",
    ),
    "test_link": ("اختبر الاتصال", "Test connection"),
    "test_warn": (
        "بنرسل 5 ثواني شاشة سودا للمنصة عشان نتأكد إن العنوان والمفتاح صحيحين. قناتك بتكون مباشرة هالثواني. نكمل؟",
        "We will send 5 seconds of a black screen to the platform to check the address and the key. Your channel is live for those seconds. Continue?",
    ),
    "test_running": ("جاري الاختبار...", "Testing..."),
    "test_ok": ("✓ المنصة قبلت البث: العنوان والمفتاح صحيحين.", "✓ The platform took the stream: the address and key are right."),
    "test_fail": ("✗ المنصة ما قبلت: {why}", "✗ The platform did not take it: {why}"),
    "test_busy": ("أوقف الحماية أول عشان تختبر.", "Stop protection first to test."),
    "out_record": ("تسجيل فيديو بس (بدون بث) - لصنّاع اليوتيوب والمقاطع", "Record a video only (no stream) - for YouTube videos and clips"),
    "record_dir": ("مجلد التسجيلات", "Recordings folder"),
    "pick_folder": ("اختر مجلد...", "Choose a folder..."),
    "record_hint": (
        "يحفظ الفيديو المحمي بجودة عالية مع الصوت في: {folder}",
        "Saves the protected video in high quality with sound in: {folder}",
    ),
    "window_hint": (
        "للي يبي يرتّب كل شي داخل OBS: يلتقط نافذة «StreamSafe Output» ويبث أو يسجل منها.",
        "For those who arrange everything in OBS: capture the «StreamSafe Output» window and stream or record from OBS.",
    ),
    "obs_guide": ("طريقة ربط OBS خطوة بخطوة", "How to connect OBS, step by step"),
    "obs_guide_text": (
        "١) في برنامجنا: اختر «من OBS» واضغط «ابدأ الحماية» أول.\n٢) في OBS: الإعدادات > البث > الخدمة: مخصص (Custom).\n٣) السيرفر: {server}  (زر «انسخ» جنبه)\n٤) مفتاح البث في OBS: خلّه فاضي.\n٥) الإعدادات > المخرجات: المعدل (Bitrate) عالي: 20000 إلى 40000 (الإرسال داخل جهازك، ما يمر على الإنترنت، فالعالي ما يضر ويحفظ الجودة)، ومعدل الإطارات الرئيسية (Keyframe) = 2.\n٦) الإعدادات > الفيديو: الدقة ومعدل الإطارات نفس اللي اخترتها في «جودة الصورة» عندنا.\n٧) اضغط «Start Streaming» في OBS. بتشوف في برنامجنا «OBS متصل ✓».\n\nمفتاح منصتك (كيك، تيك توك...) تحطه في برنامجنا مو في OBS، ونحن نرسل للمنصة بالمعدل المناسب لها. للبلايستيشن: أضف كرت الالتقاط كمصدر في OBS عادي.",
        "1) In our program: choose «From OBS» and press «Start protection» first.\n2) In OBS: Settings > Stream > Service: Custom.\n3) Server: {server}  (the «Copy» button next to it)\n4) Stream key in OBS: leave it empty.\n5) Settings > Output: a HIGH bitrate, 20000 to 40000 (this link stays inside your PC, not on the internet: high costs nothing and keeps the quality), Keyframe interval = 2.\n6) Settings > Video: the same resolution and frame rate as «Picture quality» here.\n7) Press «Start Streaming» in OBS. Our program then shows «OBS connected ✓».\n\nYour platform's key (Kick, TikTok...) goes into our program, not into OBS; we send to the platform at the bitrate it takes. For PlayStation: add the capture card as a source in OBS as usual.",
    ),
    "copy_server": ("انسخ عنوان OBS", "Copy the OBS address"),
    "lk_obs_wait": (
        "OBS: ننتظره... اضغط Start Streaming في OBS (السيرفر: {server})",
        "OBS: waiting... press Start Streaming in OBS (server: {server})",
    ),
    "lk_obs_ok": ("✓ OBS متصل والصورة توصل", "✓ OBS connected, the picture is arriving"),
    "lk_obs_lost": (
        "OBS وقف الإرسال: ننتظره يرجع ({server}). المشاهدين يشوفون شاشة الانتظار.",
        "OBS stopped sending: waiting for it to come back ({server}). Viewers see the waiting screen.",
    ),
    "lk_obs_failed": ("✗ ما قدرنا نستقبل من OBS: {why}", "✗ Could not receive from OBS: {why}"),
    "lk_in_wait": ("الدخل: جاري الفتح...", "Input: opening..."),
    "lk_in_ok": ("✓ الصورة توصل", "✓ The picture is arriving"),
    "lk_in_failed": ("✗ الدخل ما انفتح: {why}", "✗ The input did not open: {why}"),
    "lk_plat_wait": ("المنصة: نتصل أول ما تجهز أول صورة محمية", "Platform: we connect as soon as the first protected picture is ready"),
    "lk_plat_connecting": ("المنصة: جاري الاتصال...", "Platform: connecting..."),
    "lk_plat_ok": ("✓ متصل بالمنصة: البث شغال", "✓ Connected to the platform: you are live"),
    "lk_plat_lost": (
        "انقطع الاتصال بالمنصة ({why}). نعيد المحاولة بعد {s} ثانية...",
        "Lost the platform ({why}). Trying again in {s} s...",
    ),
    "lk_plat_failed": ("✗ المنصة رفضت البث: {why}", "✗ The platform refused the stream: {why}"),
    "lk_rec_wait": ("التسجيل: يبدأ أول ما تجهز أول صورة محمية", "Recording: starts with the first protected picture"),
    "lk_rec": ("● يسجل في: {file}", "● Recording to: {file}"),
    "lk_rec_done": ("✓ انحفظ التسجيل: {file}", "✓ Recording saved: {file}"),
    "lk_window": ("الخرج: نافذة «StreamSafe Output» (التقطها في OBS)", "Output: the «StreamSafe Output» window (capture it in OBS)"),
    "port_moved": (
        "المنفذ {old} ماسكه برنامج ثاني، فاستخدمنا {new}. غيّر السيرفر في OBS إلى: {server}",
        "Port {old} is used by another program, so we use {new}. Change the server in OBS to: {server}",
    ),
    "why_port_busy": ("المنفذ ماسكه برنامج ثاني", "the port is used by another program"),
    "why_no_device": ("الجهاز مو موجود أو برنامج ثاني ماسكه", "the device is missing or held by another program"),
    "why_refused_key": ("المفتاح غلط أو منتهي، أو البث مقفل من المنصة", "the key is wrong or expired, or the platform blocked the stream"),
    "why_no_route": ("ما فيه إنترنت أو العنوان غلط", "no internet, or the address is wrong"),
    "why_refused": ("السيرفر رفض الاتصال (راجع العنوان)", "the server refused the connection (check the address)"),
    "why_timeout": ("السيرفر ما رد (النت ضعيف أو محجوب)", "the server did not answer (weak or blocked internet)"),
    "why_tls": ("مشكلة في الاتصال المشفّر (rtmps)", "a problem with the encrypted connection (rtmps)"),
    "why_broken": ("الاتصال انقطع", "the connection broke"),
    "why_unknown": ("سبب غير معروف (التفاصيل في التقرير)", "unknown reason (details in the report)"),
    "why_input_ended": ("انتهى الإرسال", "the sending ended"),
    "why_cancelled": ("انلغى", "cancelled"),
    "why_no_encoder": ("ما فيه مشفّر فيديو يشتغل على جهازك", "no video encoder works on this PC"),
    "why_bad_rtmp": ("عنوان السيرفر غلط", "the server address is wrong"),
    "why_bad_rtmp_app": ("العنوان ناقص (مثلاً /app في آخره)", "the address is incomplete (for example /app at its end)"),
    "why_bad_key": ("المفتاح فاضي أو فيه مسافات", "the key is empty or has spaces"),
    "bad_rtmp_app": (
        "عنوان السيرفر ناقص: انسخه كامل من المنصة (مثلاً ينتهي بـ /app).",
        "The server address is incomplete: copy it whole from the platform (for example ending in /app).",
    ),
    "bad_input_file": ("اختر ملف الفيديو.", "Choose the video file."),
    "bad_file_output": (
        "الفيديو المسجّل يتحفظ كتسجيل بس: اختر «تسجيل فيديو بس».",
        "A recorded video can only be saved as a recording: choose «Record a video only».",
    ),
    "engine_weak_gpu": (
        "كرت الشاشة ما يكفي: ما بدأ البث (التفاصيل في تبويب الفحص).",
        "The graphics card is not enough: nothing was streamed (details in the Check tab).",
    ),
    "engine_port_busy": (
        "المنفذ ماسكه برنامج ثاني: اضغط ابدأ مرة ثانية (بنختار منفذ ثاني) وغيّر السيرفر في OBS.",
        "The port is used by another program: press Start again (we pick another one) and change the server in OBS.",
    ),
    "engine_input_failed": (
        "الدخل ما انفتح: تأكد من الجهاز أو OBS. التفاصيل في التقرير.",
        "The input did not open: check the device or OBS. Details are in the report.",
    ),
    "engine_platform_failed": (
        "المنصة رفضت البث: راجع المفتاح والعنوان (زر «اختبر الاتصال»). التفاصيل في التقرير.",
        "The platform refused the stream: check the key and address («Test connection»). Details are in the report.",
    ),
    "engine_input_ended": (
        "انتهى الدخل (OBS وقف أو خلص الفيديو). تقرير الجلسة جاهز.",
        "The input ended (OBS stopped or the video finished). The session report is ready.",
    ),
    "nvidia_needed": (
        "كرتك إنفيديا بس الوضع السريع مو مفعل، فالبرنامج بيشتغل أبطأ وممكن تكثر الشاشة الكاملة وتقطيع الغطاء. نفعّله الحين؟ (تنزيل حوالي {gb} جيجا وتجهيز حوالي 4 دقايق، مرة وحدة بس). «لا» يبدأ بالوضع البطيء.",
        "Your card is NVIDIA but the fast mode is off, so protection runs slower: more full-screen covers and choppier clothing. Turn it on now? (about {gb} GB download and about 4 minutes of preparation, once). «No» starts in the slow mode.",
    ),
    "sec_quality": ("الجودة والتأخير", "Quality and delay"),
    "resolution": ("الدقة", "Resolution"),
    "fps": ("الإطارات بالثانية", "Frames per second"),
    "quality_hint": (
        "خلّها نفس إعدادات OBS بالضبط عشان ما تقل الجودة أبداً. الفيديو المسجّل من قبل يحتفظ بدقته وإطاراته الأصلية. كيك وتويتش وتيك توك وفيسبوك ياخذون لين 1080p.",
        "Make these exactly the same as OBS so the quality never drops. A recorded video keeps its own resolution and frame rate. Kick, Twitch, TikTok and Facebook take up to 1080p.",
    ),
    "delay_auto": ("تلقائي", "Automatic"),
    "delay_hint": (
        "تلقائي = ثانيتين مع الوضع السريع لإنفيديا، و3 ثواني لغيره. الأجهزة الأضعف: 3 ثواني أحسن.",
        "Automatic = 2 s with the NVIDIA fast mode, 3 s otherwise. Weaker PCs: 3 s is better.",
    ),
    "brand_pro_only": (
        "الغطاء يحمل شعار StreamSafe. تغيير الشعار لشعارك أو شعار الراعي متاح في Pro: ادعمنا على Ko-fi ويوصلك مفتاح Pro لمدة شهر.",
        "The covers carry the StreamSafe logo. Your own or a sponsor's logo comes with Pro: support us on Ko-fi and get a one-month Pro key.",
    ),
    "brand_ours": ("شعار StreamSafe", "The StreamSafe logo"),
    "sent_bigger": (
        "⚠ OBS يرسل {sent} والحماية على {ours}: غيّر «جودة الصورة» لنفس OBS عشان ما تقل الجودة.",
        "⚠ OBS sends {sent} and protection runs at {ours}: set «Picture quality» to the same as OBS so nothing is lost.",
    ),
    "sent_low_bitrate": (
        "⚠ OBS يرسل بمعدل {kbps} فقط: ارفعه في OBS إلى 20000 أو أكثر (الإرسال داخل جهازك) عشان تبقى الصورة نظيفة.",
        "⚠ OBS sends only {kbps} kbit/s: raise it in OBS to 20000 or more (this link stays in your PC) to keep the picture clean.",
    ),
    "plat_x": ("إكس X (تويتر)", "X (Twitter)"),
    "plat_trovo": ("تروفو Trovo", "Trovo"),
    "plat_hint_x": (
        "من X: Media Studio > Producer > Sources: انسخ RTMP URL والمفتاح.",
        "On X: Media Studio > Producer > Sources: copy the RTMP URL and the key.",
    ),
    "plat_hint_trovo": (
        "من تروفو: Creator Studio > Stream Key. الصق المفتاح بس.",
        "On Trovo: Creator Studio > Stream Key. Paste the key only.",
    ),
    "sec_license": ("الاشتراك ومفتاح Pro", "Subscription and Pro key"),
    "license_key": ("المفتاح", "Key"),
    "license_activate": ("تفعيل", "Activate"),
    "license_deactivate": ("إلغاء التفعيل من هذا الجهاز", "Deactivate on this PC"),
    "license_buy": ("ادعمنا على Ko-fi", "Support us on Ko-fi"),
    "plan_basic": ("العادي", "Standard"),
    "plan_pro": ("Pro", "Pro"),
    "license_on": ("✓ الاشتراك مفعّل: {plan} (المفتاح {key})", "✓ Subscription active: {plan} (key {key})"),
    "license_grace": (
        " - بدون إنترنت الحين، يشتغل كم يوم وبعدها يحتاج يتأكد",
        " - offline now: works a few days, then it must check again",
    ),
    "license_needed": (
        "تحتاج اشتراك عشان تشغّل الحماية: الصق المفتاح اللي وصلك على الإيميل واضغط «تفعيل».",
        "Protection needs a subscription: paste the key from your e-mail and press «Activate».",
    ),
    "license_beta": (
        "البرنامج مجاني وقت الإطلاق والحماية كاملة. تبي Pro (شعارك أو شعار الراعي على الغطاء)؟ ادعمنا على Ko-fi واكتب «Pro» في الرسالة، ويوصلك على إيميلك مفتاح Pro لمدة شهر. الصقه هنا واضغط تفعيل.",
        "Free during the launch, with full protection. Want Pro (your or a sponsor's logo on the covers)? Support us on Ko-fi and write “Pro” in the message, and we email you a one-month Pro key. Paste it here and press Activate.",
    ),
    "license_deactivate_ask": (
        "نلغي تفعيل الاشتراك من هذا الجهاز عشان تقدر تفعّله في جهاز ثاني؟",
        "Deactivate the subscription on this PC so you can activate it on another?",
    ),
    "license_deactivate_offline": (
        "انحذف من هذا الجهاز، بس ما قدرنا نبلغ السيرفر (الإنترنت). لو رفض في الجهاز الثاني جرب بعدين.",
        "Removed from this PC, but the server could not be told (internet). If the other PC refuses, try again later.",
    ),
    "why_license_no_license": ("ما فيه اشتراك مفعّل (الإعدادات > الاشتراك)", "no active subscription (Settings > Subscription)"),
    "why_license_invalid": (
        "الاشتراك منتهي أو موقوف: جدّده ثم فعّل المفتاح",
        "the subscription ended or is paused: renew it, then activate the key",
    ),
    "why_license_limit": (
        "المفتاح مفعّل على أجهزة بعدد الحد: ألغِ التفعيل من جهاز ثاني أول",
        "the key is active on as many PCs as allowed: deactivate it on another PC first",
    ),
    "why_license_not_found": ("المفتاح غلط: انسخه كامل من الإيميل", "wrong key: copy it whole from the e-mail"),
    "why_license_inactive": ("المفتاح موقوف (الاشتراك منتهي أو ما انخصم)", "the key is switched off (subscription ended or unpaid)"),
    "why_license_offline_too_long": (
        "لازم إنترنت عشان نتأكد من الاشتراك (صار لك أيام بدون)",
        "internet is needed to check the subscription (it has been days)",
    ),
    "why_license_network": (
        "ما قدرنا نوصل لسيرفر الاشتراكات: تأكد من الإنترنت",
        "could not reach the subscription server: check the internet",
    ),
    "why_license_unknown_product": ("هذا المفتاح لمنتج ثاني مو لـ StreamSafe", "this key is for another product"),
    # --- updates ---
    "update_app": (
        "نسخة جديدة من البرنامج: {version}. نزّلها من الفريق وثبّتها فوق القديمة.",
        "A new version of the program: {version}. Get it from the team and install it over the old one.",
    ),
    "update_data": ("فيه تحديث للنماذج / ملفات القرار.", "There is an update for the models / decision files."),
    "update_button": ("حدّث الآن", "Update now"),
    "update_done": ("تم التحديث. النسخة السابقة محفوظة.", "Updated. The previous version is kept."),
    "update_failed": ("ما تم التحديث ({why}). تكمل بالموجود.", "The update did not happen ({why}). You continue with what you have."),
    "rollback": ("ارجع للنماذج السابقة", "Go back to the previous models"),
    "rollback_done": ("رجعنا للنسخة السابقة.", "Went back to the previous version."),
    "update_busy": ("لا يمكن التحديث أثناء الحماية.", "Cannot update while protection is running."),
    # --- misc ---
    "about": ("{name} {version}", "{name} {version}"),
    "quit_running": ("الحماية تعمل. نوقفها ونطلع؟", "Protection is running. Stop it and quit?"),
    "error": ("خطأ", "Error"),
}


def t(lang: str, key: str, **values: object) -> str:
    entry = TEXT.get(key)
    if entry is None:
        return key
    text = entry[0] if lang == "ar" else entry[1]
    text = text or entry[1 - (0 if lang == "ar" else 1)]
    try:
        return text.format(name=PRODUCT_NAME, **values)
    except (KeyError, IndexError, ValueError):
        return text


def missing() -> list[str]:
    """Keys that lack a language (used by a test)."""
    return [k for k, v in TEXT.items() if not all(v)]
