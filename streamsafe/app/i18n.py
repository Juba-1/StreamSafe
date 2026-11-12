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
    "tab_settings": ("الإعدادات", "Settings"),
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
    "install_cancelled": ("أُلغي التثبيت. ما تغير شي.", "CancelledError. Nothing was changed."),
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
    "sec_in": ("الدخل (من OBS)", "Input (from OBS)"),
    "in_srt": ("OBS يرسل عبر SRT", "OBS sends over SRT"),
    "in_camera": ("كاميرا OBS الافتراضية", "OBS Virtual Camera"),
    "srt_port": ("المنفذ", "Port"),
    "srt_hint": (
        "في OBS: الإعدادات > البث > الخدمة: مخصص، السيرفر: srt://127.0.0.1:{port}",
        "In OBS: Settings > Stream > Service: Custom, Server: srt://127.0.0.1:{port}",
    ),
    "camera_name": ("اسم الكاميرا", "Camera name"),
    "sec_out": ("الخرج (للمشاهدين)", "Output (to viewers)"),
    "out_window": ("نافذة يلتقطها OBS", "A window that OBS captures"),
    "out_rtmp": ("بث مباشر للمنصة (RTMP/RTMPS)", "Stream to the platform (RTMP/RTMPS)"),
    "rtmp_url": ("عنوان البث (بدون المفتاح)", "Stream address (without the key)"),
    "stream_key": ("مفتاح البث", "Stream key"),
    "remember_key": ("احفظ المفتاح على هذا الجهاز (محمي بحساب ويندوز)", "Remember the key on this PC (protected by your Windows account)"),
    "key_no_store": (
        "هذا الجهاز ما فيه مخزن محمي: المفتاح يبقى فقط طول ما النافذة مفتوحة.",
        "This PC has no protected store: the key stays only while the window is open.",
    ),
    "delay": ("التأخير (ثانية)", "Delay (seconds)"),
    "bad_port": ("المنفذ لازم يكون بين 1024 و 65535.", "The port must be between 1024 and 65535."),
    "bad_camera": ("اكتب اسم الكاميرا.", "Enter the camera name."),
    "bad_rtmp": ("عنوان البث لازم يبدأ بـ rtmp:// أو rtmps://", "The stream address must start with rtmp:// or rtmps://"),
    "bad_delay": ("التأخير لازم يكون بين 0.5 و 10 ثواني.", "The delay must be between 0.5 and 10 seconds."),
    "bad_key": ("اكتب مفتاح البث (بدون مسافات).", "Enter the stream key (no spaces)."),
    # --- run ---
    "start": ("ابدأ الحماية", "Start protection"),
    "stop": ("أوقف", "Stop"),
    "stopping": ("جاري الإيقاف وكتابة التقرير...", "Stopping and writing the report..."),
    "open_report": ("افتح آخر تقرير جلسة", "Open last session report"),
    "no_report": ("ما فيه تقرير بعد.", "There is no report yet."),
    "open_obs": ("إعداد OBS", "OBS setup"),
    "obs_state": ("ملف الحالة لسكربت OBS: {path}", "State file for the OBS script: {path}"),
    "obs_script": ("سكربت الحماية: {path}", "Guard script: {path}"),
    "copy": ("انسخ", "Copy"),
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
