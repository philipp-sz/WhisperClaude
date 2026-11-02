# Documentation

| File | Content |
|---|---|
| [how-it-works.md](how-it-works.md) | Technical overview: pipeline, backends, chunking, translation, keys, measurements |
| [transcription.md](transcription.md) | Backend and model choice, precision, decoding, chunking, live transcription, audio preprocessing, rejected LLM cleanup |
| [translation.md](translation.md) | Translate key, language picker, translation model comparison, language detection |
| [interface.md](interface.md) | Status pill rendering and level meter, tray icon, clipboard and history |

The measurements come from two machines:

- Laptop: Intel Core Ultra 7 258V (8 cores), Intel Arc 140V iGPU, 32 GB RAM.
- Desktop: Intel Core i5-10600K (6 cores), NVIDIA RTX 4070 (12 GB).

The quality tests use FLEURS (read speech, German and English), FLORES-200 (translation) and a corpus of 13 texts recorded by the author with `scripts/record_corpus.py`. The recordings are not part of the repository.
