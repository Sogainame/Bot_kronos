# Bot_kronos

Sniper-бот для Polymarket 5-минутных BTC-рынков с фильтром Kronos *(foundation-модель для финансовых свечей)*.

Основан на `Sniper_poly_bot` v4 + Kronos-фильтр как последний барьер перед входом в трейд.

## Архитектура

```
Terminal 1: kronos_service.py
  - Каждые 5 минут на старте окна:
    - Берёт 400 свечей BTC 5m с Binance
    - Генерирует веер 20 траекторий на MPS (~10s)
    - Пишет результат в kronos_signal.json
                |
                v (файл)
Terminal 2: bot.py (Sniper)
  - Тики цены через WebSocket
  - Сигнал от SignalEngine (delta/momentum/etc.)
  - Kronos-фильтр как ПОСЛЕДНИЙ барьер:
    - consensus < 70%       -> SKIP
    - direction = NEUTRAL   -> SKIP
    - direction расходится  -> SKIP
    - всё ОК                -> FIRE
```

**Принцип:** Kronos только блокирует плохие трейды, никогда не меняет направление бота.

## Установка (Mac Studio)

### 1. Клонируем оба репо рядом

```bash
cd ~/Desktop/Projects
git clone https://github.com/Sogainame/Bot_kronos.git
# Если Kronos ещё не склонирован:
git clone https://github.com/shiyu-coder/Kronos.git
```

Структура должна быть:
```
~/Desktop/Projects/
├── Bot_kronos/          ← этот репо
└── Kronos/              ← shiyu-coder/Kronos
```

> Если Kronos лежит в другом месте — отредактируй KRONOS_PATH в начале kronos_service.py.

### 2. Виртуальное окружение

```bash
cd ~/Desktop/Projects/Bot_kronos
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install ccxt torch transformers numpy pandas matplotlib
```

### 3. Конфиг

Скопируй свой .env от Sniper_poly_bot (где BUILDER_API_KEY, BUILDER_SECRET, BUILDER_PASSPHRASE, PK):

```bash
cp ~/path/to/old/Sniper_poly_bot/.env .
```

## Запуск (2 терминала)

### Терминал 1: Kronos-сервис

```bash
cd ~/Desktop/Projects/Bot_kronos
source venv/bin/activate
python3 kronos_service.py --device mps
```

Что увидишь:
```
[Kronos] Loading NeoQuasar/Kronos-small on mps...
[Kronos] Ready. Will publish signals to .../kronos_signal.json
[Kronos] Sleeping 142s until next window @ 15:35 UTC
[Kronos] ✓ window=1716483300 dir=UP votes=16/20 mean=+0.234% spread=0.412% (8.3s)
```

### Терминал 2: Sniper-бот (DRY режим)

```bash
cd ~/Desktop/Projects/Bot_kronos
source venv/bin/activate

# DRY (без реальных ставок) — для теста
python3 bot.py --asset btc --mode safe

# Отключить Kronos и сравнить
python3 bot.py --asset btc --mode safe --no-kronos

# LIVE (когда DRY покажет улучшение)
python3 bot.py --asset btc --mode safe --live
```

## Логика Kronos-фильтра

| Ситуация | Действие |
|---|---|
| Файла kronos_signal.json нет | Разрешить (не мешать боту) |
| Прогноз для другого окна | Разрешить (рассинхрон) |
| Прогноз старше 280 сек | Разрешить (stale) |
| Consensus < 70% (votes <=13/20) | БЛОКИРОВАТЬ (модель не уверена) |
| direction = NEUTRAL | БЛОКИРОВАТЬ |
| direction расходится с ботом | БЛОКИРОВАТЬ (модель против) |
| direction совпадает | Разрешить |

В логах бота будет виден reason:
- `READY|kronos:ok(UP@0.85)` — пропустили
- `kronos:low_consensus(0.55)` — заблокировано
- `kronos:disagree(DOWNvsUP)` — модель против

## Параметры Kronos-сервиса

```bash
python3 kronos_service.py --device mps --paths 20
```

- `--device` — mps (Apple GPU, рекомендуется), cpu, cuda
- `--paths` — сколько траекторий в веере (20 по умолчанию)
- `--model` — NeoQuasar/Kronos-small (по умолч.), можно Kronos-base (точнее, в 4x медленнее)

## Что смотреть в DRY режиме первые 20 окон

1. Сколько окон Kronos заблокировал (видно по `_last_reason` в heartbeat-логах)
2. WR без фильтра vs с фильтром — запусти параллельно `--no-kronos` и сравни
3. Inference time на MPS — должно быть 5-10 сек на 20 путей. Если 30+ — что-то не так

## Откат

```bash
python3 bot.py --asset btc --no-kronos
```

Бот работает как чистый Sniper v4 без Kronos.

## Источники

- Sniper_poly_bot v4: https://github.com/Sogainame/Sniper_poly_bot
- Kronos foundation model: https://github.com/shiyu-coder/Kronos
- Paper: https://arxiv.org/abs/2508.02739 (AAAI 2026)
