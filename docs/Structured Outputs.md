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

Схема для **медицинского отчёта** с вложенными объектами, массивами, enum и рекурсией:

```python
json_schema = {
    "format": {
        "type": "json_schema",
        "name": "medical_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "patient": {
                    "type": "object",
                    "description": "Информация о пациенте",
                    "properties": {
                        "id": {"type": "string", "format": "uuid"},
                        "name": {"type": "string"},
                        "age": {"type": "integer", "minimum": 0, "maximum": 130},
                        "gender": {"type": "string", "enum": ["male", "female", "other"]}
                    },
                    "required": ["id", "name", "age", "gender"],
                    "additionalProperties": False
                },
                "symptoms": {
                    "type": "array",
                    "description": "Перечень симптомов, которые сообщил пациент",
                    "items": {
                        "type": "string"
                    },
                    "minItems": 1
                },
                "diagnosis": {
                    "type": "object",
                    "description": "Заключение врача",
                    "properties": {
                        "has_diagnosis": {"type": "boolean"},
                        "possible_diagnoses": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "certainty": {
                            "type": "number",
                            "description": "Уровень уверенности в диагнозе (%)",
                            "minimum": 0,
                            "maximum": 100
                        }
                    },
                    "required": ["has_diagnosis", "possible_diagnoses", "certainty"],
                    "additionalProperties": False
                },
                "treatment_plan": {
                    "type": "array",
                    "description": "Пошаговый план лечения",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {"type": "integer"},
                            "description": {"type": "string"},
                            "medications": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "dosage": {"type": "string"},
                                        "duration_days": {"type": "integer"}
                                    },
                                    "required": ["name", "dosage", "duration_days"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["step_number", "description", "medications"],
                        "additionalProperties": False
                    }
                },
                "follow_up": {
                    "type": "object",
                    "description": "Рекомендации по дальнейшему наблюдению",
                    "properties": {
                        "next_visit_date": {"type": "string", "format": "date"},
                        "notes": {"type": "string"}
                    },
                    "required": ["next_visit_date", "notes"],
                    "additionalProperties": False
                }
            },
            "required": ["patient", "symptoms", "diagnosis", "treatment_plan", "follow_up"],
            "additionalProperties": False
        }
    }
}
```

---

# 🔹 Как работает процесс ответа с использованием схемы

1. **Ты формируешь запрос**
   В `responses.create` или `responses.parse` указываешь `text.format` = JSON Schema (или Pydantic модель).
   Это говорит модели: «отвечай только строго по этой структуре».

2. **Модель генерирует ответ**
   OpenAI добавляет внутренние инструкции → LLM всегда возвращает **валидный JSON** в рамках схемы.

   * Если поле `required` — оно обязательно будет.
   * Если `enum` — ответ всегда будет одним из допустимых значений.
   * Если `additionalProperties: false` — модель не «придумает» лишних ключей.
   * Если значение невозможно → модель может сделать **refusal**.

3. **SDK парсит ответ**

   * Если используешь `responses.parse` → SDK сразу возвращает объект Pydantic.
   * Если используешь `responses.create` → приходит JSON-строка, её парсишь сам.

Супер, давай разберём подробно 🔎
Каждый JSON Schema у нас кладётся в параметр `text: { format: {...} }`. Внутри этого блока есть **метаданные** (`type`, `name`, `strict`) и сама **схема** (`schema`).

---

# 🔹 Верхний уровень (`format`)

```json
{
  "format": {
    "type": "json_schema",
    "name": "medical_report",
    "strict": true,
    "schema": { ... }
  }
}
```

### Поля:

* **`type`** — всегда `"json_schema"` (говорит API, что мы хотим Structured Outputs).
* **`name`** *(обязательно)* — имя схемы, используется для идентификации в логах, дебаге, иногда в UI.
* **`strict`** *(обязательно, лучше всегда true)* — заставляет модель жёстко соблюдать схему, иначе она может «фантазировать».
* **`schema`** *(обязательно)* — сама JSON Schema (описание объекта, его свойств, ограничений).

---

# 🔹 Внутри `schema`

```json
"schema": {
  "type": "object",
  "properties": { ... },
  "required": [ ... ],
  "additionalProperties": false
}
```

### Поля:

* **`type`** *(обязательно)* — корневой тип. Всегда `"object"` (по правилам Structured Outputs root не может быть `array` или `anyOf`).
* **`properties`** *(обязательно)* — словарь с описанием всех полей.
* **`required`** *(обязательно)* — список обязательных полей. Если ключ не в этом списке → модель может его пропустить.
* **`additionalProperties`** *(обязательно, должно быть `false`)* — запрещает модельке придумывать лишние ключи (иначе может сгенерировать что-то не по схеме).

---

# 🔹 Для каждого свойства внутри `properties`

Пример:

```json
"patient": {
  "type": "object",
  "description": "Информация о пациенте",
  "properties": {
    "id": {"type": "string", "format": "uuid"},
    "age": {"type": "integer", "minimum": 0, "maximum": 130},
    "gender": {"type": "string", "enum": ["male", "female", "other"]}
  },
  "required": ["id", "age", "gender"],
  "additionalProperties": false
}
```

### Поля:

* **`type`** *(обязательно)* — базовый тип поля (`string`, `number`, `boolean`, `integer`, `array`, `object`).
* **`description`** *(необязательно, но очень полезно)* — подсказка для модели, чтобы ответы были точнее.
* **`properties`** *(обязательно для `object`)* — словарь со вложенными полями.
* **`items`** *(обязательно для `array`)* — описание типа элементов массива.
* **`required`** *(обязательно для `object`)* — список обязательных полей внутри объекта.
* **`enum`** *(необязательно, только для `string`/`number`)* — ограничивает допустимые значения.
* **`format`** *(необязательно)* — уточнение для `string` (например: `date`, `email`, `uuid`).
* **`minimum` / `maximum`** *(необязательно)* — ограничения для числовых значений.
* **`pattern`** *(необязательно)* — регулярка, которой должна соответствовать строка.
* **`minItems` / `maxItems`** *(необязательно)* — ограничения на количество элементов в массиве.
* **`additionalProperties`** *(обязательно для объектов)* — должно быть `false`, иначе Structured Outputs не сработает.

# 🔹 Обязательность полей

* В **корневой схеме** обязательны:

  * `type`
  * `properties`
  * `required`
  * `additionalProperties`

* В каждом **объекте** обязательны:

  * `type` = `"object"`
  * `properties`
  * `required`
  * `additionalProperties`

* В **массиве** обязательно:

  * `type` = `"array"`
  * `items` (тип элементов).

* В каждом поле:

  * `type` (например, `"string"`, `"boolean"`)
  * остальное (описание, ограничения) — не обязательно, но очень помогает для качества.

---

# 🔹 Зачем нужны все эти поля?

* **`description`** — влияет на качество, помогает модели не гадать.
* **`required`** — жёстко гарантирует наличие ключей.
* **`additionalProperties: false`** — защищает от "лишних полей".
* **`enum` / `format` / `pattern`** — повышают точность и снижают вероятность мусорных ответов.
* **ограничения (`minimum`, `maximum`, `minItems`)** — помогают модели не придумывать ерунду.

---

⚡ Итого:
В схеме реально обязательные для работы Structured Outputs поля:

* `type`
* `properties`
* `required`
* `additionalProperties: false`

Остальные (`description`, `enum`, `format`, `minItems`) — **рекомендованы**, чтобы получить максимально качественный и стабильный ответ.