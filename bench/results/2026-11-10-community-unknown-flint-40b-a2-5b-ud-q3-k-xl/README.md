# Community benchmark on AMD Radeon

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon RX 7900 XTX (gfx1100), 24 GiB VRAM
- CPU: AMD Ryzen 7 7800X3D 8-Core Processor (8 cores, 16 threads); installed RAM: 30.6 GiB; storage: not measured
- OS: Linux 7.3.3-1-cachyos; GPU driver: 7.3.3-1-cachyos; CUDA: not read
- PolyStrata: project 0.1.45, commit fabae0d6; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: flint-40b-a2.5b-ud-q3_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a PolyStrata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 30 (27-34) | 0 | 100 (100-100) | 9 | not measured | 120.9 (113.7-144.7) | 0.09 (0.09-0.12) |
| 4K prompt, fresh | 4100 (4100-4100) | 0 (0-0) | 24 cap | 3 | 3495.9 (3425.8-3513.3) | not measured | 1.20 (1.19-1.22) |
| 16K prompt, fresh | 16388 (16388-16388) | 0 (0-0) | 24 cap | 3 | 2315.4 (2313.7-2317.2) | not measured | 7.11 (7.10-7.11) |
| agent session, follow-up turns | 13383 (13383-13383) at the last turn | see below | up to 64 | 3 | see below | 111.4 (98.9-120.1) | 1.29 (0.69-1.59) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.3 % (68.3-68.3); earlier-turn tokens read again per session: 325 (325-325) (earlier turns that the cache did not hold; the 12605 (12605-12605) new tokens of the files are read once); first turn (nothing cached) first token: 0.21 s (0.21-0.21).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 458 | 0 | 458 | 2504.1 | 0.208 | 120.2 |
| 1 | 3548 | 457 | 3091 | 3200.8 | 0.996 | 112.5 |
| 2 | 5081 | 3547 | 1534 | 2293.0 | 0.695 | 107.6 |
| 3 | 8040 | 5080 | 2960 | 2339.6 | 1.292 | 111.4 |
| 4 | 10718 | 8039 | 2679 | 1800.4 | 1.515 | 99.3 |
| 5 | 13383 | 10717 | 2666 | 1710.0 | 1.587 | 98.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 65 s.

## Correctness checks (agent harness)

After the speed suite the same server was driven by Hermes Agent through its
OpenAI-compatible endpoint with tools enabled and reasoning off: eight tasks, each
in a fresh working directory, judged on the artifact and not on the model's text
(`hermes-agent-bench.json` in this folder has the raw data).

| Task | Wall seconds | Prompt tokens | Output tokens | Passed |
| --- | ---: | ---: | ---: | --- |
| run `date -Iseconds`, return only its output | 4.4 | 7,102 | 127 | yes |
| extract four bullets from a status file into a new file | 129.7 | 164,916 | 9,706 | yes |
| change one line in an INI file, leave the rest | 5.2 | 11,239 | 344 | yes |
| create three files with exact content | 5.4 | 11,613 | 442 | yes |
| read a missing file, create it, read it again | 4.9 | 14,949 | 342 | yes |
| return strict JSON with four fields | 4.5 | 3,540 | 349 | yes |
| summarise German text in exactly three bullets | 10.4 | 3,718 | 1,268 | yes |
| find a bug in `calc.py` until its tests pass | 7.5 | 20,282 | 577 | yes |

All eight produced the right artifact, including the tool call, the JSON and the
bug fix with passing tests. The extraction task is the outlier: the model re-read
and re-verified the file in a loop, twice (67 s / 81,503 prompt tokens and 130 s /
164,916 prompt tokens), while a 35B-A3B model on the same harness finished it in
4.1 s and 14,197 prompt tokens. Every other task stayed between 4.4 and 10.4 s.
For an agent loop that is the behaviour to watch: fast tokens, but a tendency to
keep checking instead of answering.

This model had not been run on an AMD card before (the repository lists it as
NVIDIA, and `setup` warns when a family is untried on AMD). The engine compiled
for gfx1100 without changes, tool calls and the OpenAI API worked, and this run
used the install's own configuration (`--kv-whole`, 65,536 context, one slot).
