# LLM Service

Микросервис на базе FastAPI для взаимодействия с LLM (OpenAI API) с поддержкой обычного и потокового режима ответов.
Проект спроектирован с использованием ООП и модульной архитектуры, что упрощает расширение функционала (мультимодальность, LangChain и др.).

## Возможности

* **Чат с моделью OpenAI**

  * **Обычный режим** — возвращает полный ответ целиком.
  * **Потоковый режим** — передаёт ответ частями в реальном времени, что удобно для отображения текста по мере генерации.
* **Генерация изображений** — создание одного или нескольких изображений по текстовому описанию (prompt) с возможностью настройки размера, качества и формата.
* **Транскрипция аудио** — распознавание речи из аудиофайлов в текст с поддержкой разных языков.

## Установка

### 1. Локальный запуск

**Зависимости:**

* Python 3.10+
* pip

```bash
git clone https://github.com/AntonSHBK/llm_service.git
cd llm_service
pip install -r requirements.txt
```

Создайте файл `.env` в корне проекта:

```env
OPENAI_API_KEY=ваш_api_ключ_OpenAI
```

### 2. Запуск через Docker

```bash
git clone https://github.com/AntonSHBK/llm_service.git
cd llm_service/docker
docker-compose build
docker-compose up
```

## Запуск приложения

**Локально:**

```bash
uvicorn app.main:app --reload
```
После запуска:

* Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* Проверка статуса:

```bash
curl http://127.0.0.1:8000/health
```

## Примеры API

### Чат

**Обычный чат:**

```bash
curl -X POST "http://127.0.0.1:8000/chat/text" \
-H "Content-Type: application/json" \
-d '{"input":[{"role":"user","content":"Напиши стих"}]}'
```

**Потоковый чат:**

```bash
curl -X POST "http://127.0.0.1:8000/chat/text_stream" \
-H "Content-Type: application/json" \
-d '{"input":[{"role":"user","content":"Напиши рассказ про кота"}]}'
```

---

### Аудио

**Транскрипция аудио (speech → text):**

```bash
curl -X POST "http://127.0.0.1:8000/audio/transcribe" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@test.wav"
```

**Генерация аудио (text → speech, файл):**

```bash
curl -X POST "http://127.0.0.1:8000/tts/tts_file" \
-H "Content-Type: application/json" \
-d '{"text":"Привет, это тест синтеза речи","voice":"alloy","format":"mp3"}' \
--output result.mp3
```

**Генерация аудио (text → speech, поток):**

```bash
curl -X POST "http://127.0.0.1:8000/tts/tts_bytes" \
-H "Content-Type: application/json" \
-d '{"text":"Привет, это тест синтеза речи в потоковом режиме","voice":"alloy","format":"wav"}' \
--output result.wav
```

---

### Изображения (OpenAI)

**Генерация изображения (байты → PNG):**

```bash
curl -X POST "http://127.0.0.1:8000/image/openai_bytes" \
-H "Content-Type: application/json" \
-d '{"prompt":"Иконка кота","n":1,"size":"256x256"}' \
--output cat.png
```

**Генерация изображения (файл → PNG/ZIP):**

```bash
curl -X POST "http://127.0.0.1:8000/image/openai_file" \
-H "Content-Type: application/json" \
-d '{"prompt":"Пейзаж с горами","n":2,"size":"512x512"}' \
--output images.zip
```

---

### Изображения (YandexART)

**Генерация изображения (байты → PNG):**

```bash
curl -X POST "http://127.0.0.1:8000/image/yandex_bytes" \
-H "Content-Type: application/json" \
-d '{"prompt":"Силуэт дерева","n":1,"size":"1:1"}' \
--output tree.png
```

**Генерация изображения (файл → PNG/ZIP):**

```bash
curl -X POST "http://127.0.0.1:8000/image/yandex_file" \
-H "Content-Type: application/json" \
-d '{"prompt":"Закат над морем","n":3,"size":"1:1"}' \
--output sunset.zip
```

## Запуск тестов

Убедись, что в `.env` заданы ключи:

* `OPENAI_API_KEY`
* `YANDEX_API_KEY`
* `YANDEX_FOLDER_ID`

---

### Запуск всех тестов

```bash
pytest app/tests -v
```

### Запуск тестов по папке

```bash
pytest app/tests/api -v        # только API-тесты
pytest app/tests/models -v     # только тесты моделей
```

### Запуск конкретного файла

```bash
pytest app/tests/models/test_chat.py -v
pytest app/tests/api/test_chat_api.py -v
pytest app/tests/api/test_img_api.py -v
pytest app/tests/models/test_audio.py -v
pytest app/tests/models/test_image.py -v
```

### Запуск конкретного теста внутри файла

```bash
pytest app/tests/models/test_image.py::test_openai_image_generate_bytes -v
```