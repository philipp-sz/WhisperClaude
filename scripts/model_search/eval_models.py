"""Evaluate Granite, Parakeet and Canary on the author's corpus and on FLEURS, like eval_quality.py / eval_fleurs.py.

Needs the separate environment described in README.md (torch, transformers, faster-whisper), run from there:
    python eval_models.py granite --sets corpus fleurs --noise
Work folder (model cache in hf/, FLEURS flat in fleurs/de_de and en_us): $MODEL_SEARCH_DIR, default %LOCALAPPDATA%\WhisperClaude\model_search (made by setup_env.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("MODEL_SEARCH_DIR", Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "WhisperClaude" / "model_search"))
os.environ["HF_HOME"] = str(WORK / "hf")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice")) # imported by the corpus scripts, unused here

import numpy as np # noqa: E402
import torch # noqa: E402
from faster_whisper.vad import VadOptions, get_speech_timestamps # noqa: E402

import librosa_shim # noqa: E402
from scripts.eval_fleurs import noisy, read_wav, sentences # noqa: E402
from scripts.eval_quality import REFS, clips, score # noqa: E402
from whisperclaude.textfix import join_chunks # noqa: E402

librosa_shim.install()
SR = 16_000
LEAD_IN_S = 0.5


class Granite:
    """IBM Granite Speech 5.0 470M TurboCTC (encoder only, CTC, greedy) through transformers."""

    def __init__(self, dtype: torch.dtype) -> None:
        from transformers import AutoModelForCTC, AutoProcessor
        name = "ibm-granite/granite-speech-5.0-470m-turboctc"
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForCTC.from_pretrained(name, dtype=dtype).eval()
        self.dtype = dtype

    def __call__(self, audio: np.ndarray) -> str:
        inputs = self.processor([audio], sampling_rate=SR, device=self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs.to(self.model.device, dtype=self.dtype))
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


class Parakeet:
    """NVIDIA Parakeet TDT 0.6B v3 (transducer) through transformers."""

    def __init__(self, dtype: torch.dtype) -> None:
        from transformers import AutoModelForTDT, AutoProcessor
        name = "nvidia/parakeet-tdt-0.6b-v3"
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForTDT.from_pretrained(name, dtype=dtype).eval()
        self.dtype = dtype

    def __call__(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs.to(self.model.device, dtype=self.dtype))
        return self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()


class Canary:
    """NVIDIA Canary-1B-v2 (attention encoder-decoder); the spoken language must be given in `lang`."""

    needs_language = True

    def __init__(self, dtype: torch.dtype) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        name = "nvidia/canary-1b-v2"
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(name, dtype=dtype).eval()
        self.dtype = dtype
        self.lang = "de"

    def __call__(self, audio: np.ndarray) -> str:
        inputs = self.processor.apply_transcription_request(audio=audio, source_language=self.lang)
        with torch.no_grad():
            out = self.model.generate(**inputs.to(self.model.device, dtype=self.dtype), max_new_tokens=256)
        return self.processor.decode(out, skip_special_tokens=True)[0].strip()


class Granite2B:
    """IBM Granite Speech 5.0 2B (a speech LLM, autoregressive). `keywords` biases the spelling of names."""

    skip_joined = True # the 268 s dictations would take over an hour on the CPU
    PROMPTS = {"punct": "transcribe the speech with proper punctuation and capitalization.",
               "plain": "can you transcribe the speech into a written format?"}

    def __init__(self, dtype: torch.dtype) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        name = "ibm-granite/granite-speech-5.0-2b"
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(name, dtype=dtype).eval()
        self.dtype = dtype
        self.keywords: list[str] = []
        self.prompt_kind = "punct"

    def prompt(self) -> str:
        """The user prompt: the instruction, with the keyword list if there is one."""
        text = self.PROMPTS[self.prompt_kind]
        if self.keywords:
            text = "transcribe the speech to text. Keywords: " + ", ".join(self.keywords)
        chat = [{"role": "user", "content": "<|audio|>" + text}]
        return self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def __call__(self, audio: np.ndarray) -> str:
        inputs = self.processor(self.prompt(), torch.from_numpy(audio)[None], device="cpu", return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=300, do_sample=False, num_beams=1)
        new = out[0, inputs["input_ids"].shape[-1]:].unsqueeze(0)
        return self.processor.tokenizer.batch_decode(new, add_special_tokens=False, skip_special_tokens=True)[0].strip()


MODELS = {"granite": Granite, "parakeet": Parakeet, "canary": Canary, "granite2b": Granite2B}


def chunks_of(audio: np.ndarray, max_s: float = 30) -> list[np.ndarray]:
    """VAD segments merged into chunks of <= max_s, each after 0.5 s of silence (the app's way, no language split)."""
    groups: list[list[dict]] = []
    size = 0
    for seg in get_speech_timestamps(audio, VadOptions(max_speech_duration_s=max_s)):
        n = seg["end"] - seg["start"]
        if groups and size + n <= max_s * SR:
            groups[-1].append(seg)
            size += n
        else:
            groups.append([seg])
            size = n
    lead = np.zeros(int(LEAD_IN_S * SR), np.float32)
    return [np.concatenate([lead] + [audio[s["start"]:s["end"]] for s in g]) for g in groups]


def transcribe(model, audio: np.ndarray, max_s: float = 30) -> str:
    """Text of a whole recording: chunk it like the app does, transcribe each chunk, join."""
    return join_chunks([model(chunk) for chunk in chunks_of(audio, max_s)])


def run_corpus(model, max_s: float) -> dict:
    """Transcripts and error counts of the corpus clips; joined dictations are skipped for language-bound models."""
    out = {}
    for cid, audio in clips().items():
        if cid.startswith("joined") and (getattr(model, "needs_language", False) or getattr(model, "skip_joined", False)):
            continue # several languages in one dictation, or too slow
        if getattr(model, "needs_language", False):
            model.lang = "en" if cid.startswith(("en_", "cmd_en")) else "de"
        t0 = time.perf_counter()
        out[cid] = {"text": transcribe(model, audio, max_s), "s": time.perf_counter() - t0, "audio_s": len(audio) / SR}
    sc = {c: score(REFS[c], v["text"]) for c, v in out.items()}
    errs = lambda ids: (sum(sc[i]["errors"] for i in ids if i in sc), sum(sc[i]["ref_words"] for i in ids if i in sc))
    short = [c for c in sc if not c.startswith("joined")]
    seconds = sum(out[i]["s"] for i in short)
    summary = {"single": errs([c for c in short if c.startswith(("de_", "en_", "cmd"))]),
               "mixed": errs([c for c in short if c.startswith("mix")]),
               "joined": errs([c for c in sc if c.startswith("joined")]),
               "case_errors": sum(s["case_errors"] for s in sc.values()),
               "s_per_10s": 10 * seconds / sum(out[i]["audio_s"] for i in short)}
    return {"clips": out, "summary": summary}


def run_fleurs(model, n: int, noise: bool) -> list[dict]:
    """WER on the first n digit-free FLEURS dev sentences per language, clean and optionally with 10 dB noise."""
    results = []
    for lang in ("de_de", "en_us"):
        data = [(read_wav(Path(p)), ref) for p, ref in sentences(WORK / "fleurs" / lang, n)]
        if getattr(model, "needs_language", False):
            model.lang = lang[:2]
        for kind in ("clean", "noisy") if noise else ("clean",):
            errors = words = 0
            wall = audio_s = 0.0
            for audio, ref in data:
                audio = noisy(audio) if kind == "noisy" else audio
                t0 = time.perf_counter()
                text = model(audio) if len(audio) <= 30 * SR else transcribe(model, audio)
                wall, audio_s = wall + time.perf_counter() - t0, audio_s + len(audio) / SR
                sc = score(ref, text)
                errors, words = errors + sc["errors"], words + sc["ref_words"]
            results.append({"set": lang, "kind": kind, "words": words, "errors": errors, "wer": 100 * errors / words,
                            "s_per_10s": 10 * wall / audio_s})
            print(json.dumps(results[-1]), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"]) # bf16 is very slow on this CPU
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sets", nargs="*", default=["corpus"])
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = MODELS[args.model](torch.float32 if args.dtype == "fp32" else torch.bfloat16)
    model(np.zeros(SR, np.float32))
    result: dict = {"model": args.model, "dtype": args.dtype}
    if "corpus" in args.sets:
        result["corpus"] = run_corpus(model, 30)
        print("CORPUS", json.dumps(result["corpus"]["summary"]), flush=True)
    if "fleurs" in args.sets:
        result["fleurs"] = run_fleurs(model, args.n, args.noise)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"result_{args.model}_{args.dtype}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
