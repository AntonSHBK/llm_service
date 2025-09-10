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