# translation-verifier (`mtverify`)

Verify any machine translation service in four gates: **alive**, **intact**, **accurate**, **still good**.
Works against DeepL, Google Cloud Translation, Azure Translator, or an offline file of outputs.
Ships with a 27-language smoke set and runs with no API key at all.

```
pip install -e .            # core: sacrebleu, lingua, httpx, pydantic
mtverify init-cases --out cases/smoke.jsonl
mtverify run --cases cases/smoke.jsonl --provider file
```

---

## The whole idea in one picture

```
                           YOUR TEXT
            "Hello {name}, your order #123 ships on Monday."
                                 |
                                 v
                 +-------------------------------+
                 |     TRANSLATION SERVICE       |      deepl | google | azure
                 |  (the thing you are testing)  |      file  = outputs you already have
                 +-------------------------------+
                                 |
                                 v
        "வணக்கம் {name}, உங்கள் ஆர்டர் #123 திங்கள் அன்று அனுப்பப்படும்."
                                 |
                                 v
 ==================== GATE 1  HEALTH : is it alive? ==========================
 |  one cheap call: DeepL /v2/usage, Google /languages, Azure 1-word translate |
 |  key valid? quota left? latency?                      -> pass 180 ms       |
 =============================================================================
                                 |
                                 v
 ==================== GATE 2  FUNCTIONAL : did it break anything? ============
 |  placeholders   {name} #123 still there?               -> pass 2 kept      |
 |  html_tags      <b>..</b> count and names match?       -> skip (none)      |
 |  numbers        123 still 123? (Arabic digits ok)      -> pass 1 kept      |
 |  urls_emails    links and addresses untouched?         -> skip (none)      |
 |  language       really Tamil? script + lingua          -> pass 100% Tamil  |
 |  untranslated   output is not just the input echoed?   -> pass             |
 |  length_ratio   1.37x, inside 0.3-3.5 for Tamil        -> pass             |
 |  glossary       forced terms present?                  -> skip (none)      |
 |  encoding       no U+FFFD, no control chars            -> pass             |
 |  idempotent     same input twice = same output?        -> skip (--repeat)  |
 =============================================================================
                                 |
                                 v
 ==================== GATE 3  QUALITY : does it mean the same? ===============
 |                                                                            |
 |   output  --compare-->  human reference                                    |
 |   "...திங்கள் அன்று..."     "...திங்கட்கிழமை..."                             |
 |                                                                            |
 |   chrF2  82.4  (threshold 50)                         -> pass              |
 |   BLEU   70.2  (informational)                        -> pass              |
 |   COMET  0.67  live run, --comet Unbabel/wmt22-comet-da -> FAIL (< 0.75)  |
 |   LaBSE  0.93  example value, --labse, needs no reference     (optional)  |
 =============================================================================
                                 |
                                 v
 ==================== GATE 4  MONITORING : still good next month? ============
 |  same command on a cron / CI schedule; exit code 0/1/2; reports/latest.md  |
 |  vendors change models silently. Gate 2 nightly, Gate 3 weekly on a sample |
 =============================================================================
```

### The same sentence, broken on purpose

```
  output:  "வணக்கம் பெயர், உங்கள் ஆர்டர் 123 செவ்வாய் அன்று அனுப்பப்படும்."
                       ^^^^                 ^^^           ^^^^^^^^
                 {name} translated      # dropped       Monday -> Tuesday

  GATE 2  placeholders  FAIL  missing {name}
  GATE 2  numbers       pass  (123 is still there, only the # went)
  GATE 3  chrF2 51.9    pass  (barely; the score alone would have let this through)

  verdict: FAIL.  Gate 2 caught what Gate 3 nearly missed.
```

This is why Gate 2 exists. Metric scores are averages over characters; a lost `{name}` costs a
few points but breaks the product.

---

## Quickstart, offline, no key

```
$ mtverify init-cases --out cases/smoke.jsonl
wrote cases/smoke.jsonl

$ mtverify run --cases cases/smoke.jsonl --provider file
provider file  health ok  cases 32  pass 27  fail 5  error 0  languages 27
means  chrf 92.88  bleu 89.54
failing checks  functional.placeholders=1  functional.language=1  functional.untranslated=1  quality.chrf=1
                functional.html_tags=1  functional.numbers=1  functional.glossary=1
report reports/mtverify-file-20261007T190849Z.md
$ echo $?
1
```

The smoke set has 27 good translations and 5 deliberately broken ones. Exit code 1 means
"some case failed", which is the correct answer for that file.

```
$ sed -n 1,12p reports/latest.md
# mtverify run: file

health: ok (32 hypotheses loaded from file)

| total | pass | fail | skip | error | languages | p95 ms |
|---|---|---|---|---|---|---|
| 32 | 27 | 5 | 0 | 0 | 27 | 0.0 |

| metric | mean |
|---|---|
| chrf | 92.88 |
| bleu | 89.54 |
```

## Quickstart, live provider

```
export MTVERIFY_DEEPL_KEY=...:fx            # free keys end in :fx
mtverify health --provider deepl            # Gate 1 only
mtverify run --cases cases/smoke.jsonl --provider deepl --repeat --report-dir reports
```

With a live provider the `hypothesis` field in the case file is ignored; the service output is
graded instead. `--repeat` translates every batch twice and fails any case whose two outputs differ.

---

## How a run flows through the code

```
  cases/*.jsonl ---> runner.load_cases()        validate every line first, fail before any API call
                          |
                          v
                   providers.get_provider()     file | deepl | google | azure
                          |
                          v
                   gates.health_gate()          GATE 1   fail -> exit 2, nothing else runs
                          |
                          v
                   runner._translate_all()      group by (src,tgt), batch, retry 429/5xx with backoff,
                          |                     one failing batch marks its cases "error", run continues
                          v
                   gates.functional_gate()      GATE 2   10 pure checks per case
                          |
                          v
                   gates.QualityScorer.score()  GATE 3   chrF/BLEU per case, COMET/LaBSE batched
                          |
                          v
                   report.write()               reports/<provider>-<stamp>.json + .md, latest.*
                          |
                          v
                   exit code                    0 all pass | 1 a case failed | 2 health/run error
```

```
src/mtverify/
  cli.py             argparse entry: run | health | languages | init-cases
  runner.py          orchestration, batching, retries, summary
  report.py          JSON + Markdown writers
  models.py          pydantic contracts: TestCase, CheckResult, GateResult, CaseReport, RunReport
  config.py          Settings from env vars; redacted() for logs, keys never printed
  languages.py       28-language registry, script ranges, lingua detection
  patterns.py        shared regexes: placeholders, tags, urls, numbers
  providers/         base (HttpProvider with retry), deepl, google, azure, file
  gates/             health, functional, quality
  data/smoke.jsonl   bundled 27-language smoke set
tests/               226 tests, all offline (httpx.MockTransport for providers)
```

---

## Case file format

JSON Lines, one case per line. Lines starting with `#` are comments.

```json
{"id": "ta-order", "source": "Hello {name}, your order #123 ships on Monday.",
 "source_lang": "en", "target_lang": "ta",
 "reference": "வணக்கம் {name}, உங்கள் ஆர்டர் #123 திங்கட்கிழமை அனுப்பப்படும்.",
 "hypothesis": "வணக்கம் {name}, உங்கள் ஆர்டர் #123 திங்கள் அன்று அனுப்பப்படும்.",
 "glossary": {"order": "ஆர்டர்"}, "tags": ["orders"]}
```

| field | required | meaning |
|---|---|---|
| `id` | yes | unique per file; duplicates stop the run |
| `source` | yes | text in `source_lang` |
| `source_lang` | no (default `en`) | ISO code from the registry |
| `target_lang` | yes | ISO code from the registry |
| `reference` | no | human translation; without it Gate 3 chrF/BLEU is `skip`, not `fail` |
| `hypothesis` | only for `--provider file` | the output you want graded |
| `glossary` | no | `{source term: required target term}`; each must appear in the output |
| `tags` | no | free labels, carried into the report |

Unknown language codes and malformed lines fail with the file name and line number before
any API call is made.

## Providers

| provider | env vars | health probe | notes |
|---|---|---|---|
| `file` | none | always ok | grades `hypothesis` from the case file; CI friendly |
| `deepl` | `MTVERIFY_DEEPL_KEY` | `GET /v2/usage` | `:fx` key routes to api-free.deepl.com; pt -> PT-BR, zh -> ZH-HANS |
| `google` | `MTVERIFY_GOOGLE_KEY` | `GET /languages` | Basic v2 REST, key sent as `X-goog-api-key` header so it never lands in a logged URL, `format=text` |
| `azure` | `MTVERIFY_AZURE_KEY`, `MTVERIFY_AZURE_REGION`, optional `MTVERIFY_AZURE_ENDPOINT` | one-word translate | Text Translation v3.0; zh -> zh-Hans, pt -> pt-br |

Common: `MTVERIFY_TIMEOUT` (seconds, default 30), `MTVERIFY_RETRIES` (default 3).
Retry policy: 429 and 5xx retry with exponential backoff and honour `Retry-After`; other 4xx fail at once.
Keys are read from the environment only and never appear in logs or reports (`Settings.redacted()`).

## Gate 2 checks

| check | fails when | handles |
|---|---|---|
| `not_empty` | source has text, output is blank | |
| `placeholders` | multiset of placeholders differs | `{x}` `{{x}}` `${x}` `%s` `%d` `%(x)s` `%1$s` `:x` |
| `html_tags` | tag names/closing/self-closing multiset differs | `<B>` == `<b>`, `<br/>` == `<br />` |
| `numbers` | digit groups differ | Arabic-Indic and Indic digits, `1,250.50` == `1.250,50` == `١٬٢٥٠٫٥٠` |
| `urls_emails` | any URL or email changed | |
| `language` | output is not in the target language | Unicode script ratio >= 60%, lingua for same-script pairs (hi/mr, ur/ar/fa, zh/ja) |
| `untranslated` | output equals input (case-insensitive) | skips when source is only placeholders/numbers |
| `length_ratio` | `len(out)/len(src)` outside the language's band | zh 0.15-2.0, ja/ko 0.2-2.5, others 0.3-3.5 |
| `glossary` | a required target term is missing | case-insensitive |
| `encoding` | U+FFFD or control chars present | |
| `idempotent` | second call differs (`--repeat`) | |

## Gate 3 metrics and thresholds

| metric | flag | needs reference | default threshold | read it as |
|---|---|---|---|---|
| chrF2 | always | yes | `--chrf-min 50` | character F-score; good for Tamil, Hindi and other rich morphology |
| BLEU | always | yes | `--bleu-min` (off) | word n-gram overlap; `char` tokenizer for ja/ko/th, `zh` for zh |
| COMET | `--comet MODEL` | `wmt22-comet-da` yes, `wmt22-cometkiwi-da` no | `--comet-min 0.75` | neural, trained on human ratings; the only check here that catches wrong meaning |
| LaBSE | `--labse` | no | `--labse-min 0.75` | cosine between source and output embeddings; cheap meaning check |

```
pip install -e ".[comet]"   # unbabel-comet; models download from Hugging Face on first use
pip install -e ".[labse]"   # sentence-transformers
mtverify run --cases cases/smoke.jsonl --provider deepl \
  --comet Unbabel/wmt22-cometkiwi-da --labse --gpus 1
```

If an optional package is not installed the check is `skip`. If a model fails to load the
check is `error` for that run and the other checks still complete.

Rough reading of the numbers:

| metric | weak | decent | strong |
|---|---|---|---|
| chrF2 | < 40 | 50-60 | > 60 |
| BLEU | < 20 | 30-40 | > 40 |
| COMET (wmt22-comet-da) | < 0.7 | 0.75-0.85 | > 0.9 |
| LaBSE cosine | < 0.6 | 0.7-0.8 | > 0.8 |

Compare systems only on the same case file. Absolute numbers move with the test set.

## Languages (28)

```
$ mtverify languages
code name        script      detector      bleu-tok
en   English     Latin       script+lingua 13a
ta   Tamil       Tamil       script+lingua 13a
te   Telugu      Telugu      script+lingua 13a
kn   Kannada     Kannada     script only   13a
ml   Malayalam   Malayalam   script only   13a
hi   Hindi       Devanagari  script+lingua 13a
mr   Marathi     Devanagari  script+lingua 13a
gu   Gujarati    Gujarati    script+lingua 13a
bn   Bengali     Bengali     script+lingua 13a
pa   Punjabi     Gurmukhi    script+lingua 13a
ur   Urdu        Arabic      script+lingua 13a
ar   Arabic      Arabic      script+lingua 13a
fa   Persian     Arabic      script+lingua 13a
zh   Chinese     Han         script+lingua zh
ja   Japanese    Kana+Han    script+lingua char
ko   Korean      Hangul      script+lingua char
th   Thai        Thai        script+lingua char
vi   Vietnamese  Latin       script+lingua 13a
id   Indonesian  Latin       script+lingua 13a
de   German      Latin       script+lingua 13a
fr   French      Latin       script+lingua 13a
es   Spanish     Latin       script+lingua 13a
pt   Portuguese  Latin       script+lingua 13a
it   Italian     Latin       script+lingua 13a
ru   Russian     Cyrillic    script+lingua 13a
tr   Turkish     Latin       script+lingua 13a
nl   Dutch       Latin       script+lingua 13a
pl   Polish      Latin       script+lingua 13a
```

Adding a language is one line in `languages.REGISTRY`: code, name, script, Unicode ranges,
lingua enum name (or `None`), BLEU tokenizer, length band.

## CLI

```
mtverify run --cases F --provider {file,deepl,google,azure}
             [--report-dir reports] [--chrf-min 50] [--bleu-min N]
             [--comet MODEL] [--comet-min 0.75] [--labse] [--labse-min 0.75] [--gpus 0]
             [--repeat] [--batch-size N] [--fail-fast] [-v]
mtverify health --provider {deepl,google,azure}
mtverify languages
mtverify init-cases [--out cases/smoke.jsonl]
```

| exit | meaning |
|---|---|
| 0 | health ok, every case passed |
| 1 | health ok, at least one case failed a check |
| 2 | health failed, a batch errored, or the case file is invalid |

## Gate 4: schedule it

```
# crontab: nightly Gate 1+2+3 against the live service, weekly COMET on top
0 2 * * *  cd /srv/mtverify && mtverify run --cases cases/prod.jsonl --provider deepl --repeat
0 3 * * 0  cd /srv/mtverify && mtverify run --cases cases/prod.jsonl --provider deepl --comet Unbabel/wmt22-cometkiwi-da
```

`reports/latest.json` carries `summary.by_language`, `summary.failing_checks`,
`summary.metric_means` and `summary.latency_ms_p95`, ready for any dashboard.
The GitHub Actions workflow in `.github/workflows/ci.yml` runs ruff, the tests, and the offline
smoke set on every push and pull request.

## Extending

```
new provider:  subclass HttpProvider in providers/, implement health() and translate(),
               add it to providers.PROVIDERS. Transport is injectable, so test it with
               httpx.MockTransport exactly like tests/test_providers.py.
new check:     add a pure function in gates/functional.py returning CheckResult,
               append it in functional_gate(). It must never raise.
new metric:    add a _xxx_scores() method on QualityScorer that returns one CheckResult
               per case (or None when disabled) and add it to the loop in score().
```

## Known limits

- The bundled smoke references are illustrative. Replace them with reviewed human translations
  before trusting Gate 3 scores as a release gate.
- Round-trip translation (there and back) is deliberately not a check. It cannot say which
  direction broke, so it is a weak signal.
- lingua has no Kannada or Malayalam model; those two use the script check only, which still
  catches wrong-language and untranslated output.
- COMET and LaBSE models are hundreds of MB to several GB and want a GPU for large runs.
  CometKiwi XXL needs about 44 GB of GPU memory; the base `wmt22` models fit a laptop.

## Verified

### Offline, every language

| what | how | result |
|---|---|---|
| unit + integration tests | `pytest` | 226 passed |
| per-language damage matrix | `tests/test_all_languages.py` | 27 languages x 6 cases, all caught |
| lint | `ruff check src tests` | clean |
| offline smoke run | `mtverify run --cases cases/smoke.jsonl --provider file` | 27 pass, 5 fail (the 5 planted), exit 1 |
| missing key path | `mtverify health --provider deepl` with no key | `FAIL deepl: MTVERIFY_DEEPL_KEY is not set`, exit 2 |

```
 per language (27):   good output        -> pass
                      {name} -> NAME     -> placeholders FAIL
                      123 -> 132         -> numbers FAIL
                      English echo       -> untranslated + language FAIL
                      empty              -> not_empty FAIL
                      other script       -> language FAIL
```

### Live, a real translation API

Run on 2026-10-08 against MyMemory (free public MT, no key), all 27 languages, real HTTP,
then COMET `wmt22-comet-da` on the same outputs.

```
 27 languages --> live API --> GATE 1 ok --> GATE 2 --> GATE 3 chrF + COMET
                  p95 790 ms                 |            |
                                             |            +-- lowest COMET: mr 0.60 ur 0.63 kn 0.64 ta 0.67
                                             |                "ships" came back as "ships (boats)"
                                             +-- gu: {name} came back as {NAME}  -> FAIL
```

| layer | caught |
|---|---|
| Gate 2 rules | gu `{NAME}` placeholder change |
| chrF | nothing new; boat errors still scored 53-58 |
| COMET | all 4 boat errors ranked bottom, below 0.75; clean European outputs 0.91-0.97 |

Lesson: rules catch broken structure, only COMET catches wrong meaning. Turn it on for release gates.

Sources for the metric guidance: [Unbabel COMET](https://github.com/Unbabel/COMET),
[GEMBA-MQM](https://arxiv.org/abs/2310.13988), [MetricX-24](https://arxiv.org/abs/2410.03983),
[sacrebleu](https://github.com/mjpost/sacrebleu), [lingua-py](https://github.com/pemistahl/lingua-py).
