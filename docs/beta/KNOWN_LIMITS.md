# StreamSafe beta: known limits / الحدود المعروفة

This is what we know does not work or is not proven yet. If you hit something not on this list, tell us.
هذي الأشياء اللي نعرف إنها ما تشتغل أو ما انثبتت بعد. لو واجهت شي مو هنا، قل لنا.

## English

1. **Beta accuracy.** The checks can miss a character (too little cover) or cover things that are fine (too much cover). We do not have final numbers yet. Do not rely on it alone for a public stream; use a private test stream, and use your own moderation and panic key.
2. **Consoles are not supported.** The tested path is PC only. A console + capture card flow exists in the design, but it is not tested and not supported in this beta.
3. **A PC graphics card is needed.** No graphics card, no start (it never falls back to the CPU). NVIDIA RTX 3060 or better is recommended. AMD and Intel run slower and were measured less. We have not tested every card; the memory and speed thresholds in `check` are estimates.
4. **Running the game on the same PC.** StreamSafe uses the GPU too, so the game and StreamSafe compete for it. We have not measured this. Expect lower game FPS, or more full-screen slates, on weaker cards.
5. **Latency.** The picture reaches viewers about 2 seconds late (plus the platform's own delay). Chat will feel late. A shorter delay is possible but is not tested for safety.
6. **Setup A mutes the mic with the game.** In the SRT setup OBS sends one mixed sound track (game + mic). StreamSafe delays it with the picture and silences all of it while a full-screen cover is on. Use setup B if the mic must stay on.
7. **Setup B needs a careful OBS setup** (hidden scene, virtual camera, script, sync offsets). It is easy to get something wrong; use the 10-second kill test and a private stream.
8. **Fixed output quality.** The stream out of StreamSafe is 30 frames per second, H.264 at 6 Mbps, 16:9. You cannot change this in the beta.
9. **Windows only.** Live streaming is supported on Windows 10/11.
10. **Installer and window are not final.** The installer, model download, and a graphical window are being built. Until then you use the Command Prompt as in the guide.
11. **No report on Ctrl+C.** The session report is written only when a session ends by itself.
12. **Slate at start and on trouble.** For a moment at start, and whenever the PC is too slow or something errors, viewers see the full-screen slate. This is intended.
13. **On-screen game prompts (HUD reading) are off by default** and not part of this beta test.
14. **The name and look may change.**

## العربي

1. **الدقة تجريبية.** ممكن يفوّت شخصية (تغطية ناقصة) أو يغطي شي عادي (تغطية زايدة). ما عندنا أرقام نهائية. لا تعتمد عليه لحاله في بث عام؛ جرّب على بث خاص، وخلّ عندك مشرفين وزر الطوارئ.
2. **الكونسول غير مدعوم.** المسار المختبَر هو الكمبيوتر فقط. فكرة الكونسول مع كرت التقاط موجودة في التصميم لكنها ما انختبرت وما هي مدعومة في هالنسخة.
3. **لازم كرت شاشة في الكمبيوتر.** بدون كرت ما يشتغل (ما يرجع للمعالج أبدًا). الأفضل NVIDIA RTX 3060 أو أعلى. كروت AMD وIntel أبطأ وقياساتها أقل. ما جرّبنا كل الكروت؛ حدود الذاكرة والسرعة في `check` تقديرية.
4. **تشغيل اللعبة على نفس الجهاز.** StreamSafe يستخدم كرت الشاشة برضو، فاللعبة والبرنامج يتنافسون عليه. ما قسنا هذا. توقع نزول في فريمات اللعبة أو شاشات تغطية أكثر على الكروت الأضعف.
5. **التأخير.** الصورة توصل المشاهدين متأخرة حوالي ثانيتين (غير تأخير المنصة). الشات بيحسّونه متأخر. تأخير أقصر ممكن لكنه ما انختبر من ناحية الأمان.
6. **الطريقة A تكتم المايك مع اللعبة.** في طريقة SRT يرسل OBS مسار صوت واحد مخلوط (اللعبة + المايك). StreamSafe يأخّره مع الصورة ويكتمه كله طول ما التغطية الكاملة موجودة. لو لازم المايك يبقى شغال استخدم الطريقة B.
7. **الطريقة B تحتاج ضبط دقيق في OBS** (مشهد مخفي، كاميرا افتراضية، سكربت، تأخير الصوت). سهل تغلط فيها؛ سوّ اختبار القتل (Task Manager) في ثواني وجرّب على بث خاص.
8. **جودة الإخراج ثابتة.** البث الخارج من StreamSafe 30 فريم بالثانية، H.264 بسرعة 6 ميجابت، بنسبة 16:9. ما تقدر تغيّرها في النسخة التجريبية.
9. **ويندوز فقط.** البث المباشر مدعوم على ويندوز 10 و11.
10. **المثبّت والنافذة مو نهائيين.** المثبّت وتنزيل النماذج ونافذة رسومية قيد البناء. إلى ذلك الوقت تستخدم Command Prompt مثل الدليل.
11. **ما يُكتب تقرير عند Ctrl+C.** التقرير ينكتب فقط لما تنتهي الجلسة لوحدها.
12. **شاشة التغطية عند البداية وعند المشاكل.** لحظة عند البداية، ومتى ما كان الجهاز بطيء أو صار خطأ، المشاهدين يشوفون شاشة التغطية الكاملة. هذا مقصود.
13. **قراءة رسائل اللعبة على الشاشة (HUD) مطفية افتراضيًا** وليست جزءًا من هذا الاختبار.
14. **الاسم والشكل ممكن يتغيّرون.**
