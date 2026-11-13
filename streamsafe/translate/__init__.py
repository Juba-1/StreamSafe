"""Live English -> Arabic translation of the text on the screen (subtitles and prompts), drawn on the output
AFTER the cover is applied. Off by default; see ``docs/TRANSLATE.md``.

Safety contract (tests/test_translate_*.py check each line):
  * everything slow (reading the text, translating, shaping the Arabic) runs in its own process behind a
    bounded queue; the frame loop only copies a small crop into that queue and, when it shows a frame, blends a
    small finished picture. A slow, failing or dead worker means no Arabic, never a late or uncovered frame;
  * the Arabic is only drawn on a frame the gate composed normally (never on the slate, the scene card or a held
    frame), and never changes the gate's own decisions, mute or analysis;
  * the text of a frame is read from that frame and shown on the same frame's output time (the delay buffer),
    so viewers see the Arabic when the English appears.
"""
