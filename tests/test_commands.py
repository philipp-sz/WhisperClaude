"""Spoken formatting commands: pure text in, text out. ('⏎' in the tables stands for a line break.)"""
import pytest

from whisperclaude.commands import COMMANDS, apply_commands, prompt_examples


def nl(s: str) -> str:
    return s.replace("⏎", "\n")


# ---- what Whisper actually writes (turbo and small, pauses and flow) -> expected result ----
@pytest.mark.parametrize("raw, expected", [
    # new line
    ("Hello team. Command new line. Here is the update.", "Hello team.⏎Here is the update."),
    ("Hello team command new line here is the update", "Hello team⏎here is the update"),
    ("Hello team, command new line, here is the update.", "Hello team⏎here is the update."),
    ("Hello Team Command New Line here is the update.", "Hello Team⏎here is the update."),
    ("Hello team. Commandnew line. Here is the update.", "Hello team.⏎Here is the update."),
    ("Hello team command newline here is the update", "Hello team⏎here is the update"),
    ("Hello team command new-line here", "Hello team⏎here"),
    # new paragraph
    ("Thanks for the call. Command new paragraph. Here are the next steps.",
     "Thanks for the call.⏎⏎Here are the next steps."),
    ("Thanks for the call command new paragraph here are the next steps",
     "Thanks for the call⏎⏎here are the next steps"),
    ("One command new line command new paragraph two", "One⏎⏎two"),
    # bullets
    ("My shopping list command bullet milk command bullet eggs command bullet bread",
     "My shopping list⏎- milk⏎- eggs⏎- bread"),
    ("My shopping list, command bullet milk, command bullet eggs, command bullet bread.",
     "My shopping list⏎- milk⏎- eggs⏎- bread"),
    ("My list. Command bullet milk. Command bullet eggs. Command new paragraph. That is all.",
     "My list.⏎- milk⏎- eggs⏎⏎That is all."),
    ("Agenda command new paragraph command bullet budget command bullet hiring command new paragraph that is all",
     "Agenda⏎⏎- budget⏎- hiring⏎⏎that is all"),
    ("Command bullet first item", "- first item"),
    ("Command bullet point first item", "- first item"),
    # quotes
    ("He said command open quote this is fine command close quote and then he left",
     'He said "this is fine" and then he left'),
    ("He said, command open quote, this is fine, command close quote, and then he left.",
     'He said, "this is fine", and then he left.'),
    ("He said command open quote this is fine command close quote.", 'He said "this is fine".'),
    ("She wrote Command Open Quote hello world Command Close Quote in the file.",
     'She wrote "hello world" in the file.'),
    ("It was command opening quote nice command closed quote he said", 'It was "nice" he said'),
    # Whisper puts periods/commas after the command words when you pause around them
    ("Anna said. Command open quote. We should not rush. Command close quote. Then he left.",
     'Anna said. "We should not rush." Then he left.'),
    ("Anna said, command open quote, we should not rush, command close quote, then he left.",
     'Anna said, "we should not rush", then he left.'),
    ("He said command open quote. Hello. Command close quote.", 'He said "Hello."'),
    # a bullet item ends at the end of its sentence; following text starts after a blank line
    ("My list. Command bullet milk. Command bullet eggs. That is all.",
     "My list.⏎- milk⏎- eggs⏎⏎That is all."),
    ("Reminders. Command bullet call the bank. Then I go home.",
     "Reminders.⏎- call the bank⏎⏎Then I go home."),
    ("Shopping command bullet buy milk, eggs and bread. Command bullet pay the bill. After that we leave.",
     "Shopping⏎- buy milk, eggs and bread⏎- pay the bill⏎⏎After that we leave."),
    ("Shopping command bullet why is it so expensive? Because of the tax.",
     "Shopping⏎- why is it so expensive?⏎⏎Because of the tax."),
    ("Notes command bullet he said command open quote hi. there command close quote. Then more.",
     'Notes⏎- he said "hi. there"⏎⏎Then more.'),  # a sentence end inside quotes doesn't end the item
    ("Command bullet one command bullet two", "- one⏎- two"),                       # flow: ends at the next command
    ("Intro. Command bullet one. Command new line. Text", "Intro.⏎- one⏎Text"),     # explicit command wins
    # separator
    ("Part one is done. Command separator. Part two starts here.",
     "Part one is done.⏎⏎---⏎⏎Part two starts here."),
    ("Part one is done command separator part two starts here",
     "Part one is done⏎⏎---⏎⏎part two starts here"),
    ("Hello, command separator, there", "Hello⏎⏎---⏎⏎there"),
    ("Command separator hello", "---⏎⏎hello"),
    ("Hello command separator", "Hello⏎⏎---⏎⏎"),
    ("A line command new paragraph command separator command new paragraph B line",
     "A line⏎⏎---⏎⏎B line"),
    # alone / at the edges: the line break is kept (dictating only the command pastes a line break)
    ("Command new line", "⏎"),
    ("Command new paragraph", "⏎⏎"),
    ("Hello command new line", "Hello⏎"),
    ("Command new line hello", "⏎hello"),
])
def test_commands(raw, expected):
    assert apply_commands(raw) == nl(expected)


# ---- sentences that merely contain the words: must come out exactly as they went in ----
@pytest.mark.parametrize("text", [
    "I added a new line to the config file.",
    "Please start a new paragraph after the introduction.",
    "The bullet point on slide 3 is too long.",
    "Can you send me a quote for the new laptop?",
    "He opened the quote and closed the file.",
    "The hash table needs a new line of code.",
    "Strip the hash, new lines and spaces from the file.",
    "Please print the hash. New line characters are removed.",
    "Use this command. New line characters are removed.",
    "The command line is fast.",
    "Run the commands new line by line.",
    "This command bullet-proofs the build.",
    "Recommand new line.",
    "Command banana new line.",
    "The command bullets are listed below.",
    "Every command, bullet and quote is documented.",
    # Whisper dropped the comma in the real test run, so "command bullet" looked like a command:
    "Each command bullet point and quote is explained in the manual.",
    "Use the command bullet by bullet to review the plan.",
    "The command new line feed is the old name for it.",
    "This command bullet by bullet is slow.",
    "Der Command new line wird später erklärt.",
    "Your command open quote handling is broken.",
    "The separator between the columns is two pixels wide.",
    "Use a command separator in the settings file.",
    "This command separator is documented.",
    "We need a command for a new paragraph.",
    "Ich habe eine neue Zeile in die Datei eingefügt.",
    "Bitte beginne nach der Einleitung einen neuen Absatz.",
    "Das Zitat am Anfang des Buches ist sehr bekannt.",
    "",
    "Just a normal sentence.",
])
def test_ordinary_text_is_untouched(text):
    assert apply_commands(text) == text


@pytest.mark.parametrize("text", [
    "Hello team, command, new line, here is the update.",   # comma between the two words
    "Hello team. Command. New line. Here is the update.",   # period between the two words
])
def test_separated_trigger_is_a_miss_not_a_guess(text):
    """If Whisper puts punctuation between the words they are left as text: visible and easy
    to fix, while accepting it would also turn real sentences into commands."""
    assert apply_commands(text) == text


def test_period_between_trigger_and_command_is_not_a_command():
    """"Command. New line." is a sentence boundary, not a command (a long pause may cause
    this; the words then stay visible, which beats silently changing normal text)."""
    text = "Use this command. New line characters are removed."
    assert apply_commands(text) == text


def test_custom_trigger_and_bullet():
    assert apply_commands("List format bullet one format bullet two", trigger="format", bullet="* ") \
        == "List\n* one\n* two"
    assert apply_commands("List command bullet one", trigger="format") == "List command bullet one"
    assert apply_commands("one please format new line two", trigger="please format") == "one\ntwo"


def test_bullet_item_period_is_dropped_only_for_bullet_lines():
    out = apply_commands("Intro sentence. Command new line. Command bullet milk. Command bullet eggs and bread.")
    assert out == "Intro sentence.\n- milk\n- eggs and bread"


def test_idempotent_on_its_own_output():
    out = apply_commands("A command new paragraph command bullet b command bullet c")
    assert apply_commands(out) == out


def test_prompt_examples_cover_every_command():
    examples = prompt_examples()
    for c in COMMANDS:
        assert f"Command {c}." in examples
    assert prompt_examples("format").startswith(" Format new line.")


def test_noun_guard_does_not_block_real_commands():
    """Only a determiner/possessive DIRECTLY before the trigger blocks it."""
    assert apply_commands("My list command bullet milk command bullet eggs") == "My list\n- milk\n- eggs"
    assert apply_commands("Remember the milk. Command new line. Thanks.") == "Remember the milk.\nThanks."
    assert apply_commands("It was that. Command bullet next") == "It was that.\n- next"  # sentence boundary
