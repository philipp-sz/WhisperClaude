"""Record the quality test corpus: read each text aloud, press Enter, the next one starts.

.venv\\Scripts\\python.exe scripts\\record_corpus.py          # record (resumes where you stopped)
.venv\\Scripts\\python.exe scripts\\record_corpus.py --redo de_einigen   # record one text again

-> bench_audio/corpus/<id>.wav (git-ignored: your voice is never committed). The texts are the
reference for scripts/eval_quality.py. They target errors seen in real use: similar-sounding
words (einigen/reinigen), German with English loanwords switching to English, an invented
"command ..." in German speech; plus mixed speech and spoken commands, which must keep working.
"""
import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.benchmark import AUDIO_DIR, SR, save_wav  # noqa: E402
from whisperclaude.recorder import Recorder  # noqa: E402

CORPUS_DIR = AUDIO_DIR / "corpus"

CORPUS = [  # (id, text) -- generic content only, no personal details
    ("de_einigen",
     "Hey Tom, ich glaube, wir sollten uns bis Freitag auf einen Termin einigen. Die Küche muss "
     "vorher noch gereinigt werden, das übernehme ich. Kannst du mir kurz Bescheid geben, ob "
     "Samstagvormittag bei dir passt? Ansonsten würde ich Sonntag vorschlagen."),
    ("de_email",
     "Sehr geehrte Frau Becker, vielen Dank für Ihre schnelle Rückmeldung. Anbei sende ich Ihnen "
     "die überarbeitete Version des Berichts. Ich habe das Feedback aus dem letzten Meeting "
     "eingearbeitet und die Deadline im Projektplan angepasst. Falls noch Fragen offen sind, "
     "können wir gerne nächste Woche telefonieren. Mit freundlichen Grüßen"),
    ("de_anglizismen",
     "Notiz für morgen: Den Pull Request für das neue Feature reviewen, die Tests laufen lassen "
     "und danach das Release vorbereiten. Außerdem muss ich das Update für den Laptop "
     "installieren, weil der Akku seit dem letzten Update viel schneller leer ist."),
    ("de_uni",
     "Die Vorlesung fängt am Dienstag um Viertel nach zehn an. Bis dahin sollten wir die ersten "
     "drei Übungsblätter bearbeitet haben. Ich habe mir vorgenommen, jeden Tag ungefähr eine "
     "Stunde zu lernen, damit ich am Ende nicht alles auf einmal machen muss."),
    ("de_alltag",
     "Ich habe mich wieder einmal gefragt, ob es sich lohnt, das alte Fahrrad reparieren zu "
     "lassen. Seit Wochen steht es im Keller, und eigentlich wäre es Zeit, eine Entscheidung zu "
     "treffen. Vielleicht verkaufe ich es einfach und kaufe mir ein gebrauchtes."),
    ("en_message",
     "Hi Sarah, thanks for sending the slides so quickly. I had a look at them this morning and "
     "I think the structure works really well. Could you add one more example to the third "
     "section? Otherwise I'm happy with it, and we can send it to the team tomorrow."),
    ("en_prompt",
     "Can you refactor this function so that it reads the config file only once? Right now it "
     "opens the file every time it's called, which is slow. Please keep the type hints, add a "
     "short docstring, and write a pytest test that checks the default values."),
    ("en_email",
     "Dear Mr. Miller, I'm writing to ask whether the deadline for the application can be "
     "extended by one week. I'm still waiting for one of my certificates, which should arrive "
     "by the end of the month. I would really appreciate your understanding. Best regards"),
    ("en_notes",
     "Things to do this week: buy groceries, call the landlord about the heating, and finish "
     "the second chapter of the book. I also want to go running at least three times, ideally "
     "in the morning before it gets too warm."),
    ("mix_claude",
     "Ich habe gestern mit Claude an meinem Projekt gearbeitet, and honestly it went much better "
     "than I expected. Der Code ist jetzt viel übersichtlicher, and the tests finally pass. "
     "Morgen schaue ich mir noch die Performance an."),
    ("mix_paper",
     "Kurze Frage zu dem Paper: Do you think the results in table two are significant? Ich bin "
     "mir nicht ganz sicher, ob die Stichprobe groß genug ist, but maybe I'm overthinking it."),
    ("cmd_de",
     "Einkaufsliste command new line command bullet Milch command bullet Brot command bullet "
     "Äpfel command new paragraph Und bitte denk daran, die Pfandflaschen mitzunehmen."),
    ("cmd_en",
     "Hi Anna, command new paragraph thanks for your message. He said command open quote we'll "
     "see command close quote and left. command new paragraph Best, Tom"),
    # Long dictations (~80 s): several chunks; [Pause] = stop talking for ~3 s, mid-sentence on
    # purpose (that's where the VAD cuts, and where language switches were seen).
    ("long_de",
     "Hallo zusammen, kurzes Update zum Projekt. Wir haben uns gestern im Meeting darauf "
     "geeinigt, dass wir das Release um eine Woche verschieben, weil noch zwei wichtige Bugs "
     "offen sind und wir das Feedback von den ersten Testern noch nicht komplett eingearbeitet "
     "haben. [Pause] Das heißt konkret, dass der neue Termin jetzt der vierzehnte ist und "
     "nicht mehr der siebte. Ich weiß, dass das für einige von euch nicht ideal ist, aber ich "
     "denke, es ist besser, wenn wir etwas Stabiles ausliefern, als wenn wir danach wieder "
     "alles reparieren müssen. [Pause] Bis Mittwoch wäre es super, wenn jeder einmal "
     "durchgeht, welche Aufgaben bei ihm noch offen sind, und das kurz im Ticket vermerkt. "
     "Falls jemand Hilfe braucht oder sich nicht sicher ist, wie er weitermachen soll, meldet "
     "euch einfach bei mir, dann finden wir eine Lösung. Danke euch und bis morgen."),
    ("long_mix",
     "Ich wollte dir noch kurz erzählen, wie es mit meinem Projekt läuft. Ich habe die letzten "
     "Tage viel mit dem Training experimentiert, and I think I finally found the problem with "
     "the learning rate. [Pause] Es lag gar nicht am Optimizer, sondern daran, dass der "
     "Scheduler nach dem Warmup viel zu schnell abgefallen ist. Now the loss goes down "
     "smoothly, and the evaluation looks much more stable than before. [Pause] Als Nächstes "
     "möchte ich das Modell ein bisschen größer machen und schauen, ob sich das lohnt, but I "
     "need to be careful with the memory, because my laptop only has so much. Wenn du "
     "magst, können wir am Wochenende mal zusammen draufschauen, I would really appreciate a "
     "second opinion."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", nargs="*", default=[], help="ids to record again")
    args = ap.parse_args()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(i, t) for i, t in CORPUS
            if i in args.redo or not (CORPUS_DIR / f"{i}.wav").exists()]
    if not todo:
        sys.exit(f"All {len(CORPUS)} texts are recorded ({CORPUS_DIR}). Use --redo <id> to redo.")

    print(f"\n{len(todo)} short texts, about {len(todo) * 30 // 60 + 1} minutes.\n"
          "Read each text aloud in your normal dictation voice (mistakes are fine, just keep\n"
          "going), then press Enter: the next text appears and records right away.\n"
          "  r + Enter = record this text again     q + Enter = stop (resume later)\n"
          "Say 'command ...' exactly where the text has it. At [Pause], stop talking for\n"
          "about 3 seconds, then continue.\n")
    input("Press Enter to start...")
    rec = Recorder()
    k = 0
    while k < len(todo):
        cid, text = todo[k]
        print("\033[2J\033[H", end="")  # clear the screen: only the current text
        print(f"[{k + 1}/{len(todo)}]\n\n" + textwrap.fill(text, 80) + "\n")
        rec.start()
        cmd = input("● REC  read it aloud, then Enter   (r = again, q = stop): ").strip().lower()
        audio = rec.stop()
        if cmd == "q":
            break
        if cmd == "r":
            continue
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if len(audio) < 2 * SR or peak < 0.02:
            input(f"That was {'very short' if len(audio) < 2 * SR else 'silent (mic muted?)'}"
                  " - press Enter to record this text again...")
            continue
        save_wav(CORPUS_DIR / f"{cid}.wav", audio)
        k += 1
    done = sum((CORPUS_DIR / f"{i}.wav").exists() for i, _ in CORPUS)
    print(f"\n{done}/{len(CORPUS)} texts recorded in {CORPUS_DIR}. Thanks!")


if __name__ == "__main__":
    main()
