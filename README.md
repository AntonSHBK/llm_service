# LLM Microservice

Микросервис на базе FastAPI для взаимодействия с LLM (OpenAI API) с поддержкой обычного и потокового режима ответов.
Проект спроектирован с использованием ООП и модульной архитектуры, что упрощает расширение функционала (мультимодальность, LangChain и др.).

## Возможности

* Чат с моделью OpenAI:

  * Обычный режим — возвращает полный ответ.
  * Потоковый режим — передаёт ответ частями в реальном времени.
* Поддержка генерации изображений и транскрипции аудио (API-роуты можно подключить дополнительно).
* Логирование с ротацией файлов.
* Контроль количества токенов.
* Готовая структура для интеграции LangChain.

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

**Обычный чат:**

```bash
curl -X POST "http://127.0.0.1:8000/chat/" \
-H "Content-Type: application/json" \
-d '{"messages":[{"role":"user","content":"Напиши стих"}]}'
```

**Потоковый чат:**

```bash
curl -X POST "http://127.0.0.1:8000/chat/stream" \
-H "Content-Type: application/json" \
-d '{"messages":[{"role":"user","content":"Напиши рассказ про кота"}]}'
```

## Запуск тестов

Убедитесь, что в `.env` задан `OPENAI_API_KEY`.

**Запуск всех тестов:**

```bash
pytest app/test_api -v
```

**Запуск конкретного теста:**

```bash
pytest app/tests/test_api.py -v
```
