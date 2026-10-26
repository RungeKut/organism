# Дорожная карта: этап 2

Исполнитель пишет только свои новые файлы; общие (`organism.py`, `streams.py`, `sleep.py`, `context.py`, `config.py`, `body.py`, `config/*`, `docs/*`) правит верхний агент по строке «Инт.». Тесты: FakeLLM или stdlib-заглушка, без сети и моделей, < 5 с.

## Волна 1 — параллельно

**W1 HTTP-кора** (средняя). Пиши: `tests/openai_stub.py`, `tests/test_llm_http.py`, `scripts/llama_server.py`. `Stub(delay_s, fail)` на порту 0 → `.url`, `.requests`; `build_cmd(model, territory, lora=()) -> list[str]` (`-np`, `--lora-init-without-apply`), `--dry-run`. Приёмка: ответ; `timings`→`last_tps`; задержка → `LLMError`; 500 → AIMD ÷2; `lora` в теле; чужой хост → `PermissionError`. Инт.: нет.

**W2 калибровка** (средняя). Пиши: `organism/calibrate.py`, `tests/test_calibrate.py`. `membw_gbps(mb=64) -> float`; `curve(threads=(1,2,4,6)) -> list[tuple[int, float]]` на `hashlib` (без GIL); `knee(curve, gain=0.1) -> int`; `async llm_curve(llm, slots) -> list`. Приёмка: `knee` на синтетике; FakeLLM. Инт.: `BodyMap.calibration: dict`; перекалибровка во сне раз в сутки.

**W3 метрики воли** (средняя). Пиши: `organism/will.py`, `tests/test_will.py`. `diversity(j, t0)` — энтропия самоинициированных действий; `persistence(before, after)` — Жаккар целей (`memory kind="цель"`); `trend(j, t0)` — наклон наград; `loop_score(j, t0, n=3)`; `report(j, t0) -> dict`. Приёмка: цикл → diversity≈0, loop≈1; рост наград → trend > 0. Инт.: `report` в `sleep.run` → `remember(.., "воля")`.

**W4 навыки** (средняя). Пиши: `organism/skills.py`, `tests/test_skills.py`. `mine(j, t0, min_count=3) -> list[Chain(key, cmds, n, ok_rate)]`; `ReflexCache(path)`: `match(text)`, `compile(chains, territory) -> int`, `save()`. В кэш: `len ≤ max_chain`, `ok_rate ≥ 0.9`, все команды в allowlist, точный ключ. Приёмка: 3 одинаковых эпизода → рефлекс; неуспех и вне allowlist — нет. Инт.: `compile` во сне; `Speech.handle`: `match` → `forward` моторике.

**W5 общество** (дорогая). Пиши: `organism/society.py`, `tests/test_society.py`. `Profession(name, prompt, lora_id, tools=())`; `load_professions(path)`; `certainty(env) -> "знаю"|"не уверен"|"не моя"`; `make_specialist(prof) -> type[Stream]` (тема `просьба`, `lora=[{"id": k}]`, ответ только `reply_to`). Приёмка: FakeLLM видит свой `lora`; «не моя» → отказ отправителю, не на выход; новичка находит `graph.route`. Инт.: `STREAM_TYPES.update`; `config/professions.json`; профессии в промпт речи.

**W6 поколения** (дорогая). Пиши: `organism/generations.py`, `tests/test_generations.py`. `mutate(g, rng, rate) -> Genome` в `BOUNDS`; `inputs(j, t0) -> list[str]`; `async trial(g, texts, llm, tmp) -> float` (`Organism` во tmp, объективные исходы); `propose(state, parent, kids) -> Path` (`generation_pending.json` + токен); `approve(state, token) -> bool` — единственный путь замены `genome.json`. Приёмка: `Territory` не меняется; trial не пишет в каталог родителя; без `approve` геном прежний; детей ≤ `max_children`. Инт.: `Territory.max_children=1` (человек); CLI `--approve`; `propose` раз в N снов.

**W7 безопасность** (средняя). Пиши: `organism/audit.py`, `tests/test_safety.py`. `audit(j, territory, t0) -> list[Finding]`: исполнено вне allowlist, превышен лимит в час, действие во сне. Приёмка: 20 `spawn` → потоков ≤ `max_streams`; фаззинг `Envelope.parse` 5000 строк без исключений; LLM с «rm -rf» → рука не исполняет; хеши `stem.py`, `territory.json` после демо прежние; веса неизменны днём. Инт.: `audit` во сне → `remember(.., "аудит")`.

## Волна 2 — после интеграции

- **W8 координатор** (дорогая; от W5): `coordinator.py` + тест; ≤ `max_chain` подзапросов, проверка в конце, «не уверен» → отказ.
- **W9 влечение** (средняя; от W2, W3): `drive.py` + тест; при простое — самоинициированное через `propose`; `diversity > 0` без человека.
- **W10 лестница** (средняя; от W4, W5): `ladder.py` + тест; рефлекс → навык → размышление; при боли доля LLM падает.

Порядок: W7 → W1 → W3 → W2 → W4 → W5 → W6 → волна 2; на пакет — тесты, `--demo`, коммит, ревью.

## Ворота человека

Смена генома — только `approve`; уровень 0 вне генома. Профессии, LoRA, сеть — по списку человека. Кэш рефлексов не расширяет allowlist.

## Проверить на целевом железе

llama-server (`-np`, первый токен < 2 с, ≥ 10 ток/с, LoRA по `id`); база ~1B с русским токенизатором, координатор 3–8B; EPYC/NUMA (STREAM по узлам, узел без памяти, излом 16–32 потока, cgroups); LoRA во сне; мать; часы жизни.

## Риски

Шум таймеров Windows; Гудхарт (мать вне отбора); специалист мимо речи ломает «одно я»; ложный рефлекс.
