# Apollo

Репозиторий: модель эмоций в диалогах (MELD, IEMOCAP и др.). Исходный код Python — в каталоге **`back/`** (`dataset/`, `models/`, `app/`, `benchmarks/`). Статический веб-интерфейс — в **`front/`**.

- **Документация, схемы, картинки:** [documentation/README.md](documentation/README.md)
- **Артефакты обучения (чекпоинты, логи):** каталог `results/` — см. [results/README.md](results/README.md)
- **Лицензия:** [LICENSE](LICENSE)

Быстрый старт после клонирования: `pip install -r requirements.txt`. Для импортов `dataset`, `models`, `app` задайте **`PYTHONPATH`** на каталог `back/` (например, из корня репозитория: `export PYTHONPATH=back`, затем команды вида `python -m ...` или переход в `cd back` и `export PYTHONPATH=.`).

Веб-интерфейс Eleos: из корня `PYTHONPATH=back uvicorn app.main:app --host 127.0.0.1 --port 8765`.
