from typing import TypedDict, Optional, List


class SchemaBuilder:
    # https://platform.openai.com/docs/guides/structured-outputs?example=chain-of-thought#refusals
    """
    Вспомогательный класс для построения JSON Schema в формате,
    совместимом с OpenAI Structured Outputs.

    Позволяет быстро описывать поля и объекты без ручного написания dict.

    Поддерживаемые типы данных (как константы):
        - SchemaBuilder.STRING
        - SchemaBuilder.NUMBER
        - SchemaBuilder.INTEGER
        - SchemaBuilder.BOOLEAN
        - SchemaBuilder.OBJECT
        - SchemaBuilder.ARRAY

    Пример использования:
    ---------------------
    >>> schema = SchemaBuilder.object(
    ...     name="user",
    ...     properties={
    ...         "id": SchemaBuilder.field(
    ...             SchemaBuilder.STRING,
    ...             description="Уникальный идентификатор пользователя",
    ...             format_="uuid"
    ...         ),
    ...         "age": SchemaBuilder.field(
    ...             SchemaBuilder.INTEGER,
    ...             description="Возраст пользователя",
    ...             minimum=0,
    ...             maximum=120
    ...         ),
    ...         "is_active": SchemaBuilder.field(
    ...             SchemaBuilder.BOOLEAN,
    ...             description="Флаг активности пользователя"
    ...         ),
    ...         "tags": SchemaBuilder.field(
    ...             SchemaBuilder.ARRAY,
    ...             description="Список тегов пользователя",
    ...             items=SchemaBuilder.field(SchemaBuilder.STRING)
    ...         )
    ...     },
    ...     required=["id", "age", "is_active"]
    ... )
    >>> import json
    >>> print(json.dumps(schema, indent=2, ensure_ascii=False))
    {
      "format": {
        "type": "json_schema",
        "name": "user",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "id": {
              "type": "string",
              "description": "Уникальный идентификатор пользователя",
              "format": "uuid"
            },
            "age": {
              "type": "integer",
              "description": "Возраст пользователя",
              "minimum": 0,
              "maximum": 120
            },
            "is_active": {
              "type": "boolean",
              "description": "Флаг активности пользователя"
            },
            "tags": {
              "type": "array",
              "description": "Список тегов пользователя",
              "items": {
                "type": "string"
              }
            }
          },
          "required": ["id", "age", "is_active"],
          "additionalProperties": false
        }
      }
    }
    """
    
    STRING: str = "string"
    NUMBER: str = "number"
    INTEGER: str = "integer"
    BOOLEAN: str = "boolean"
    OBJECT: str = "object"
    ARRAY: str = "array"
    
    DATE: str = "date"
    DATETIME: str = "date-time"
    UUID: str = "uuid"
    EMAIL: str = "email"
    
    @staticmethod
    def field(
        type_: str,
        description: str | None = None,
        enum: list[str] | None = None,
        items: dict | None = None,
        format_: str | None = None,
        pattern: str | None = None,
        minimum: int | float | None = None,
        maximum: int | float | None = None,
        min_items: int | None = None,
        max_items: int | None = None,
    ) -> dict:
        """
        Генерация описания отдельного поля JSON Schema.

        :param type: базовый тип ("string", "number", "integer", "boolean", "object", "array")
        :param description: описание поля (для повышения качества ответов модели)
        :param enum: список допустимых значений
        :param items: схема элементов массива (если type = "array")
        :param format_: формат строки (например: "date", "uuid", "email")
        :param pattern: регулярка, которой должна соответствовать строка
        :param minimum: минимальное числовое значение
        :param maximum: максимальное числовое значение
        :param min_items: минимальное количество элементов в массиве
        :param max_items: максимальное количество элементов в массиве
        """
        field: dict = {"type": type_}

        if description:
            field["description"] = description
        if enum:
            field["enum"] = enum
        if items:
            field["items"] = items
        if format_:
            field["format"] = format_
        if pattern:
            field["pattern"] = pattern
        if minimum is not None:
            field["minimum"] = minimum
        if maximum is not None:
            field["maximum"] = maximum
        if min_items is not None:
            field["minItems"] = min_items
        if max_items is not None:
            field["maxItems"] = max_items

        return field

    @staticmethod
    def object(
        name: str,
        properties: dict,
        required: list[str],
        strict: bool = True,
    ) -> dict:
        """
        Генерация JSON Schema объекта верхнего уровня.

        :param name: название схемы
        :param properties: словарь полей
        :param required: список обязательных полей
        :param strict: жёсткий режим Structured Outputs (лучше всегда True)
        """
        return {
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": strict,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        }


class SimpleField(TypedDict, total=False):
    """Типизация для простого поля"""
    name: str
    type: str
    description: Optional[str]
    required: bool
    enum: Optional[List[str]]
    format: Optional[str]


def get_simple_json_schema(
    name: str,
    fields: List[SimpleField],
    strict: bool = True
) -> dict:
    """
    Быстрое построение JSON Schema без вложенных структур.

    :param name: название схемы
    :param fields: список словарей с описанием полей
                   [
                     {"name": "id", "type": SchemaBuilder.STRING, "description": "User ID", "required": True},
                     {"name": "age", "type": SchemaBuilder.INTEGER, "required": False}
                   ]
    :param strict: жёсткий режим Structured Outputs
    :return: dict-схема
    """
    properties = {}
    required = []

    for field in fields:
        properties[field["name"]] = SchemaBuilder.field(
            type_=field["type"],
            description=field.get("description"),
            enum=field.get("enum"),
            format_=field.get("format")
        )
        if field.get("required", False):
            required.append(field["name"])

    return SchemaBuilder.object(
        name=name,
        properties=properties,
        required=required,
        strict=strict
    )
