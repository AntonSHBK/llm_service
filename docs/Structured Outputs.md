## 1. JSON Mode

### Описание

JSON Mode — базовый режим работы, при котором модель гарантирует, что результат будет валидным JSON-объектом.
Это полезно, если нужно просто получить JSON-ответ без строгого контроля за структурой.

### Особенности

* Обеспечивает корректный синтаксис JSON.
* Не проверяет соответствие каким-либо схемам.
* Не гарантирует наличие всех нужных ключей или корректные типы значений.
* Поддерживается большинством моделей (включая `gpt-3.5-turbo`, `gpt-4-turbo`).

### Пример (Python)

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-3.5-turbo-0125",
    input=[
        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
        {"role": "user", "content": "Who won the world series in 2020? Return in format {winner: ...}"}
    ],
    text={"format": {"type": "json_object"}}
)

print(response.output_text)
# Пример результата: {"winner": "Los Angeles Dodgers"}
```

---

## 2. Structured Outputs

### Описание

Structured Outputs — расширение JSON Mode, которое обеспечивает:

* строгое соблюдение заданной **JSON Schema**;
* отказ от добавления лишних ключей;
* надёжную типизацию (строки, числа, массивы, перечисления);
* возможность использовать Pydantic (Python) или Zod (JS) для описания схемы.

Этот режим позволяет быть уверенным, что модель вернёт именно ту структуру, которую ожидает приложение.

### Преимущества

* Типобезопасность: исключается вероятность “сломанных” ответов.
* Явные отказы: если модель не может ответить из-за ограничений безопасности, в ответе будет поле `refusal`.
* Универсальность: можно описывать вложенные структуры, рекурсию, ограничения по типам.

### Пример (Python, Pydantic)

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."}
    ],
    text_format=CalendarEvent,
)

print(response.output_parsed)
# Пример результата:
# CalendarEvent(name="science fair", date="Friday", participants=["Alice", "Bob"])
```

### Пример (JSON Schema напрямую)

```python
response = client.responses.create(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract event details."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."}
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "calendar_event",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "date", "participants"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)
print(response.output_text)
```

---

## 3. Function Calling

### Описание

Function Calling используется, когда модель должна **вызывать внешние функции** или API.
Вместо того чтобы возвращать только JSON, модель выбирает функцию и передаёт аргументы в строгом формате.
Это полезно для интеграции модели с базами данных, API, внешними инструментами.

### Преимущества

* Возможность управлять приложением через LLM.
* Чёткая валидация аргументов по JSON Schema.
* Подходит для агентов, чат-ботов и интеграции с внешними сервисами.

### Пример (Python)

```python
from openai import OpenAI

client = OpenAI()

functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["C", "F"]}
            },
            "required": ["location", "unit"]
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "What's the weather like in Paris today in Celsius?"}
    ],
    functions=functions,
    function_call="auto"
)

print(response.choices[0].message)
# Пример ответа:
# {
#   "role": "assistant",
#   "function_call": {
#     "name": "get_weather",
#     "arguments": {
#       "location": "Paris",
#       "unit": "C"
#     }
#   }
# }
```

---

## 4. Сравнение подходов

| Подход             | Назначение                                       | Проверка структуры | Где использовать                                                                 |
| ------------------ | ------------------------------------------------ | ------------------ | -------------------------------------------------------------------------------- |
| JSON Mode          | Валидный JSON без строгой схемы                  | Нет                | Простые сценарии, где достаточно корректного JSON                                |
| Structured Outputs | Жёсткое соблюдение JSON Schema, типобезопасность | Да                 | Генерация данных под UI, извлечение информации, формирование отчётов             |
| Function Calling   | Модель вызывает функцию и передаёт параметры     | Да                 | Интеграция с API, базы данных, внешние сервисы, построение агентов и ассистентов |

## Пример: схема данных о пользователе и его заказах

```json
{
  "name": "user_profile",
  "description": "Информация о пользователе и его заказах",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string",
        "description": "Уникальный идентификатор пользователя (например UUID)"
      },
      "name": {
        "type": "string",
        "description": "Полное имя пользователя"
      },
      "email": {
        "type": "string",
        "format": "email",
        "description": "Электронная почта пользователя"
      },
      "age": {
        "type": "integer",
        "minimum": 0,
        "maximum": 120,
        "description": "Возраст пользователя"
      },
      "is_active": {
        "type": "boolean",
        "description": "Флаг активности пользователя"
      },
      "orders": {
        "type": "array",
        "description": "Список заказов пользователя",
        "items": {
          "type": "object",
          "properties": {
            "order_id": {
              "type": "string",
              "description": "Уникальный номер заказа"
            },
            "product": {
              "type": "string",
              "description": "Название товара"
            },
            "quantity": {
              "type": "integer",
              "minimum": 1,
              "description": "Количество товаров в заказе"
            },
            "price": {
              "type": "number",
              "minimum": 0,
              "description": "Цена за единицу товара в USD"
            },
            "status": {
              "type": "string",
              "enum": ["pending", "paid", "shipped", "delivered", "cancelled"],
              "description": "Статус заказа"
            }
          },
          "required": ["order_id", "product", "quantity", "price", "status"],
          "additionalProperties": false
        }
      }
    },
    "required": ["id", "name", "email", "is_active", "orders"],
    "additionalProperties": false
  }
}
```

---

## Объяснение ключевых частей

1. **Корневая структура**

   ```json
   {
     "name": "user_profile",
     "description": "Информация о пользователе и его заказах",
     "strict": true,
     "schema": { ... }
   }
   ```

   * `name` — имя схемы (произвольное, но лучше осмысленное).
   * `description` — описание для разработчиков.
   * `strict` — указывает, что модель должна строго следовать схеме.

2. **Основной объект**

   ```json
   "type": "object",
   "properties": { ... }
   ```

   Говорим, что верхний уровень — объект с набором свойств.

3. **Простые поля**

   * `id`: строка (UUID).
   * `name`: строка (имя).
   * `email`: строка, дополнительно проверяется как `email`.
   * `age`: число в диапазоне `0–120`.
   * `is_active`: булево значение.

4. **Массив заказов**

   ```json
   "orders": {
     "type": "array",
     "items": { "type": "object", ... }
   }
   ```

   Это массив объектов, где каждый объект — заказ.

5. **Вложенный объект заказа**

   * `order_id`: строка (номер заказа).
   * `product`: строка (название товара).
   * `quantity`: целое число ≥ 1.
   * `price`: число ≥ 0.
   * `status`: строка, но только из перечисления `pending | paid | shipped | delivered | cancelled`.

6. **Ограничения**

   * `required`: список обязательных полей.
   * `additionalProperties: false`: запрещает добавление лишних ключей, не описанных в схеме.

---

## Как можно использовать

* Для **структурированного вывода модели** (например, генерация профилей клиентов).
* Для **валидации входных данных** от пользователя.
* Для **UI-рендеринга** (на основе схемы можно построить формы).

