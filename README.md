# PCB Defect Detection

Этот репозиторий содержит исходный код, используемый для предварительной обработки, обучения, оценки и визуализации результатов модели YOLOv11. В частности, этот исходный код используется для обнаружения дефектов на печатных платах (PCB). 


## Структура Репозитория
- `src/` содержит исходный код проекта:
    - `pcb/`: содержит исходный код для предварительной обработки, обучения, оценки, визуализации, утилит
    - `scripts/`: одержит скрипты для выполнения анализа
- `.gitignore`: содержит файлы и папки, которые игнорируются git
- `.python-version`: содержит документацию для проекта
- `README.md`: содержит документацию для проекта
- `poetry.lock`: содержит файл блокировки для проекта
- `pyproject.toml`: содержит метаданные проекта и зависимости

## Установка
1. Клонируйте репозиторий
2. Перейдите в директорию `cd PCB_defect_detection`
2. Используйте python v3.10 (если вы используете [pyenv](https://github.com/pyenv/pyenv), run `pyenv local 3.10.0`)
2. Установите зависимости с помощью [poetry v1.8](https://python-poetry.org/)

```bash
poetry install
```

ПРИМЕЧАНИЕ: Если вы используете GPU, poetry автоматически устанавливает версию PyTorch для GPU (при условии, что система на linux. Если вы используете другую систему, вам может потребоваться вручную установить правильную версию PyTorch или отредактировать настройки в файле[`pyproject.toml`](pyproject.toml) file.)


## Данные
Официальный источник [`YOLO11`](https://docs.ultralytics.com/ru/models/yolo11/#what-are-the-key-improvements-in-ultralytics-yolo11-compared-to-previous-versions)
Скрипты предполагают, что [`PCB_DATASET`](https://www.kaggle.com/datasets/akhatova/pcb-defects/data) хранится в папке на том же уровне, что и репозиторий, т.е. когда вы выполняете 
`ls ..` , он покажет что-то вроде этого:
```
PCB_DATASET/
PCB_defect_detection/
```

## Запуск Анализа
1. Предварительная обработка данных
```bash
poetry run yolo-preprocess
```
Это создаст папку `output` внутри папки `PCB_DATASET/` со следующей структурой:
```
├── background
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```

2. Обучение модели
```bash
poetry run yolo-train
```
Это запустит обучение YOLOv11 на наборе данных PCB. Модель будет сохранена в папке `pcb_YOLOv8n_all_epochs_{$EPOCHS}_batch_{$BATCHES}/train/` внутри `PCB_defect_detection/`.
Выполнение `ls ..` должно показать следующую структуру:
```
├── PCB_DATASET
├── PCB_defect_detection
└── results
    └── weights
```

3. Оценка модели
```bash
poetry run yolo-train-result
```
Это оценит модель на тестовом наборе, скопирует выход модели из
`pcb_YOLOv8n_all_epochs_{$EPOCHS}_batch_{$BATCHES}/train/` в папку `results/` и создаст один примерный график выходных данных модели.


4. Прогнозирование модели
```bash
poetry run yolo-predict
```
Запускает инференс на тестовом наборе данных и сохраняет результаты в папке `results/predict/`.
Выполнение `ls ..` должно показать следующую структуру:
```tree
├── presentation
└── results
    ├── predict
    │   └── labels
    └── weights
```
