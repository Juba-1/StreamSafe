# StreamSafe beta: known limits / الحدود المعروفة

This is what we know does not work or is not proven yet. If you hit something not on this list, tell us.
هذي الأشياء اللي نعرف إنها ما تشتغل أو ما انثبتت بعد. لو واجهت شي مو هنا، قل لنا.

## English
1. **Beta accuracy.** The checks can miss a character (too little cover) or cover things that are fine (too much cover). We do not have final numbers yet. Do not rely on it alone for a public stream; use a private test stream, and use your own moderation and panic key.
2. **Consoles are not supported.** The tested path is PC only. A console + capture card flow exists in the design, but it is not tested and not supported in this beta.
3. **A PC graphics card is needed.** No graphics card, no start (it never falls back to the CPU). NVIDIA RTX 3060 or better is recommended. AMD and Intel run slower and were measured less. We have not tested every card; the memory and speed thresholds in the Check tab are estimates.
4. **Running the game on the same PC.** StreamSafe uses the GPU too, so the game and StreamSafe compete for it. We have not measured this. Expect lower game FPS, or more full-screen slates, on weaker cards.
5. **Latency.** The picture reaches viewers about 2 seconds late (plus the platform's own delay). Chat will feel late. A shorter delay is possible but is not tested for safety.
6. **Sound in setup A is new.** In the SRT setup OBS sends one mixed sound track (game + mic). StreamSafe now passes it on, delayed with the picture, and silences all of it while a full-screen cover is on, so the mic is muted too. Sync is by design but little tested: do the clap test on a private stream and tell us. Use setup B if the mic must stay on.
7. **Setup B needs a careful OBS setup** (hidden scene, virtual camera, script, sync offsets). It is easy to get something wrong; use the 10-second kill test and a private stream.
8. **Fixed output quality.** The stream out of StreamSafe is 30 frames per second, H.264 at 6 Mbps, 16:9. You cannot change this in the beta.
9. **Windows only.** Live streaming is supported on Windows 10/11.
10. **The installer is not code-signed.** Windows shows "Windows protected your PC" (More info, then Run anyway), and some antivirus programs may warn about a new unsigned program. We have not scanned it with every vendor. Tell us if yours blocks it.
11. **The beta expires.** Each build stops starting streams on the date shown on the Home tab (30 days after it was built). A stream already running is not cut. Install the new version we send. The clock check is a courtesy, not a licence.
12. **First run downloads a lot.** The models are about 3.4 GB once, and the fast NVIDIA mode about 3.5 GB more plus about 4 minutes of preparation. The model download may not be public yet; then use the zip or folder we send.
13. **The window is new.** Some texts in the Check tab (card names, error details) may be in English even in Arabic. Settings are saved when you press Start protection.
14. **No report after a crash or a kill.** The session report is written when you press Stop, close the window, or end the console version with Ctrl+C. If the program crashes or you kill it in Task Manager, there may be none.
15. **Slate at start and on trouble.** For a moment at start, and whenever the PC is too slow or something errors, viewers see the full-screen slate. This is intended.
16. **On-screen game prompts (HUD reading) are off by default** and not part of this beta test.
17. **The name and look may change.**
18. **Arabic translation of on-screen text is experimental and off by default** (Settings tab). It reads the subtitles at the bottom of the picture (and nothing else) and draws a dark Arabic box over them on the stream only, never in your own game. It runs beside the protection and can never delay or uncover a frame; if it is slow or fails you simply get no Arabic. The translation model is small and makes mistakes (it can drop or garble words, names and slang); it only reads white game-style subtitles; it is not drawn on a full-screen cover. Switching it on for the first time downloads about 96 MB. See `docs/TRANSLATE.md`.

## العربي
1. **الدقة تجريبية.** ممكن يفوّت شخصية (تغطية ناقصة) أو يغطي شي عادي (تغطية زايدة). ما عندنا أرقام نهائية. لا تعتمد عليه لحاله في بث عام؛ جرّب على بث خاص، وخلّ عندك مشرفين وزر الطوارئ.
2. **الكونسول غير مدعوم.** المسار المختبَر هو الكمبيوتر فقط. فكرة الكونسول مع كرت التقاط موجودة في التصميم لكنها ما انختبرت وما هي مدعومة في هالنسخة.
3. **لازم كرت شاشة في الكمبيوتر.** بدون كرت ما يشتغل (ما يرجع للمعالج أبدًا). الأفضل NVIDIA RTX 3060 أو أعلى. كروت AMD وIntel أبطأ وقياساتها أقل. ما جرّبنا كل الكروت؛ حدود الذاكرة والسرعة في تبويب الفحص تقديرية.
4. **تشغيل اللعبة على نفس الجهاز.** StreamSafe يستخدم كرت الشاشة برضو، فاللعبة والبرنامج يتنافسون عليه. ما قسنا هذا. توقع نزول في فريمات اللعبة أو شاشات تغطية أكثر على الكروت الأضعف.
5. **التأخير.** الصورة توصل المشاهدين متأخرة حوالي ثانيتين (غير تأخير المنصة). الشات بيحسّونه متأخر. تأخير أقصر ممكن لكنه ما انختبر من ناحية الأمان.
6. **الصوت في الطريقة A جديد.** في طريقة SRT يرسل OBS مسار صوت واحد مخلوط (اللعبة + المايك). StreamSafe الحين يمرّره، مأخّر مع الصورة، ويكتمه كله طول ما التغطية الكاملة موجودة، يعني المايك ينكتم معه. التزامن مقصود لكنه ما انختبر كثير: سوّ اختبار التصفيقة على بث خاص وقل لنا. لو لازم المايك يبقى شغال استخدم الطريقة B.
7. **الطريقة B تحتاج ضبط دقيق في OBS** (مشهد مخفي، كاميرا افتراضية، سكربت، تأخير الصوت). سهل تغلط فيها؛ سوّ اختبار القتل (Task Manager) في ثواني وجرّب على بث خاص.
8. **جودة الإخراج ثابتة.** البث الخارج من StreamSafe 30 فريم بالثانية، H.264 بسرعة 6 ميجابت، بنسبة 16:9. ما تقدر تغيّرها في النسخة التجريبية.
9. **ويندوز فقط.** البث المباشر مدعوم على ويندوز 10 و11.
10. **المثبّت غير موقّع رقميًا.** ويندوز يطلع "Windows protected your PC" (More info ثم Run anyway)، وبعض برامج الحماية ممكن تحذّر من برنامج جديد غير موقّع. ما فحصناه عند كل الشركات. لو حجبه برنامجك قل لنا.
11. **النسخة التجريبية تنتهي.** كل نسخة تبطّل تبدأ بث بالتاريخ اللي يظهر في تبويب الرئيسية (بعد 30 يوم من بنائها). البث الشغّال ما ينقطع. ثبّت النسخة الجديدة اللي نرسلها. فحص التاريخ مجرد تنظيم مو ترخيص.
12. **أول تشغيل ينزّل كثير.** النماذج حوالي 3.4 جيجا مرة وحدة، والوضع السريع لإنفيديا حوالي 3.5 جيجا زيادة مع حوالي 4 دقايق تجهيز. ممكن تنزيل النماذج ما يكون مفتوح للعامة بعد؛ وقتها استخدم الـ zip أو المجلد اللي نرسله.
13. **النافذة جديدة.** بعض النصوص في تبويب الفحص (أسماء الكروت وتفاصيل الأخطاء) ممكن تطلع بالإنجليزي حتى في العربي. الإعدادات تنحفظ لما تضغط ابدأ الحماية.
14. **ما يُكتب تقرير لو البرنامج انهار أو انقتل.** التقرير ينكتب لما تضغط أوقف، أو تسكّر النافذة، أو توقف نسخة الأوامر بـ Ctrl+C. لو البرنامج انهار أو قتلته من Task Manager ممكن ما يكون فيه تقرير.
15. **شاشة التغطية عند البداية وعند المشاكل.** لحظة عند البداية، ومتى ما كان الجهاز بطيء أو صار خطأ، المشاهدين يشوفون شاشة التغطية الكاملة. هذا مقصود.
16. **قراءة رسائل اللعبة على الشاشة (HUD) مطفية افتراضيًا** وليست جزءًا من هذا الاختبار.
17. **الاسم والشكل ممكن يتغيّرون.**
18. **ترجمة النصوص على الشاشة للعربي تجريبية ومطفية افتراضيًا** (تبويب الإعدادات). تقرأ الترجمات (السبتايتل) اللي تحت الصورة فقط وترسم فوقها صندوق عربي غامق في البث فقط، مو في لعبتك. تشتغل جنب الحماية وما تقدر تأخّر فريم ولا تكشفه؛ لو بطيئة أو صار فيها خطأ ما يطلع عربي وبس. نموذج الترجمة صغير ويغلط (ممكن يطيّح كلمات أو يخربط الأسماء والعامية)، ويقرأ فقط السبتايتل الأبيض بشكل ألعاب، وما تُرسم فوق شاشة التغطية الكاملة. أول تفعيل ينزّل حوالي 96 ميجا. التفاصيل في `docs/TRANSLATE.md`.
