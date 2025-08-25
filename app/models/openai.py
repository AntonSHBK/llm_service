import base64
import io
from pathlib import Path
from typing import Optional, Any, Union, BinaryIO, IO, Literal

from openai import OpenAI, APIError
from openai._streaming import Stream
from openai.types.responses import Response, ParsedResponse

from app.models.base_model import (
    TextGanerateModel,
    AudioGenerateModel,
    AudioTranscribeModel,
    ImageGenerateModel
)
from app.services.token_manager import TokenManager


class OpenAITextModel(TextGanerateModel):

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str = "gpt-5-nano",
        max_tokens: Optional[int] = 2048,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"Chat модель OpenAI инициализирован с моделью {self.model_name}")

    def _validate_token_limit(self, messages: list[dict[str, str]]) -> None:
        if self.max_tokens is None:
            self.logger.debug("Лимит токенов не установлен — проверка пропущена.")
            return
        total_tokens = TokenManager.count_message_tokens(messages, self.model_name)
        if total_tokens > self.max_tokens:
            self.logger.warning(f"Превышен лимит токенов ({total_tokens}/{self.max_tokens})")
            raise ValueError(f"Превышен лимит токенов: {total_tokens}/{self.max_tokens}")
           
    def get_reasoning(
        self,
        effort: Literal["minimal", "low", "medium", "high"] = "minimal",
        summary: Optional[Literal["auto", "concise", "detailed"]] = None,
        generate_summary: Optional[Literal["auto", "concise", "detailed"]] = None,
    ) -> dict[str, str]:
        """
        Вспомогательный метод для формирования reasoning-конфига.

        :param effort: уровень усилий при рассуждениях
        :param summary: режим вывода резюме рассуждений (auto | concise | detailed)
        :param generate_summary: deprecated, оставлено для совместимости
        :return: словарь конфигурации reasoning
        """
        reasoning: dict[str, str] = {"effort": effort}
        if summary is not None:
            reasoning["summary"] = summary
        if generate_summary is not None:
            reasoning["generate_summary"] = generate_summary
        self.logger.debug(f"Сформирован reasoning-конфиг: {reasoning}")
        return reasoning
    
    def generate_with_reason(
        self,
        inputs: Union[str, list[dict[str, str]]],
        model_name: str = "gpt-5-nano",
        reasoning: dict[str, str] = {"effort": "low"},
        **kwargs
    ) -> Response:
        """
        Генерация текста с указанием уровня размышления. Работает только с моделями GPT-5.
        :param inputs: строка или список сообщений (dict с role/content)
        :param reasoning: "low" | "medium" | "high"
        """
        try:
            resp: Response = self.client.responses.create(
                model=model_name,
                input=inputs,
                reasoning=reasoning,
                **kwargs
            )
            self.logger.debug(f"Ответ модели (reasoning={reasoning}): {resp.output_text}")
            return resp
        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API в generate_reason: {e}")
            raise

    def generate(
        self, 
        input: list[dict[str, str]], 
        model_name: str = "gpt-5-nano", 
        **kwargs
    ) -> Response:
        # https://platform.openai.com/docs/api-reference/chat/create
        self._validate_token_limit(input)
        try:
            resp: Response = self.client.responses.create(
                model=model_name,
                input=input,
                **kwargs
            )
            self.logger.debug(f"Ответ модели: {resp.output_text}")
            return resp

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время generate(): {e}")
            raise
        
    def generate_with_shema(
        self,
        input: list[dict[str, str]],
        text_format: Any,
        model_name: str = "gpt-5-nano",
        **kwargs
    ) -> ParsedResponse:
        self._validate_token_limit(input)
        try:
            resp: ParsedResponse = self.client.responses.parse(
                model=model_name,
                input=input,
                text_format=text_format,
                **kwargs
            )
            self.logger.debug(f"Ответ модели с разбором в схему {text_format}: {resp.output_parsed}")
            return resp

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время generate_with_shema(): {e}")
            raise
      
    def generate_stream(
        self, 
        input: list[dict[str, str]], 
        model_name: str = "gpt-5-nano", 
        **kwargs
    ) -> Stream:
        """
        Пример:
        
        stream = llm.generate_stream(input=input)

        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="")

        """
        # https://platform.openai.com/docs/api-reference/chat/create
        self._validate_token_limit(input)
        try:
            stream = self.client.responses.create(
                model=model_name,
                input=input,
                stream=True,
                **kwargs
            )
            return stream

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время generate_stream(): {e}")
            raise
    
    def complete_input(
        self, 
        role: Literal["system", "developer", "user", "assistant", "tool"], 
        context: str
    ) -> dict[str, str]:
        """
        Вспомогательный метод для формирования сообщения.

        :param role: роль сообщения 
                     ("system", "developer", "user", "assistant", "tool")
        :param context: текст сообщения
        :return: словарь в формате {"role": role, "content": context}
        """
        message = {"role": role, "content": context}
        return message
    
    def reusable_prompt(self):
        """
         Можно создать некий промт с переменными и его использовать
        """
        # https://platform.openai.com/docs/guides/text#message-roles-and-instruction-following
        pass
    
    def build_json_shema(self):
        """
        Параметр при генерации называется text_format (pedantic) и text для ручного описания
        
        Пример pedantic:
        from openai import OpenAI
        from pydantic import BaseModel

        client = OpenAI()

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]         
        """
        # https://platform.openai.com/docs/guides/structured-outputs?example=chain-of-thought#refusals
        pass
    

class OpenAITranscribeModel(AudioTranscribeModel):

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str = "gpt-4o-mini-transcribe",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"Audio модель OpenAI инициализирована с моделью {self.model_name}")
      
    def generate(
        self,
        audio_source: Union[str, Path, io.BytesIO],
        filename: Optional[str] = None,
        **kwargs
    ) -> str:
        
        try:
            if isinstance(audio_source, (str, Path)):
                file_obj = open(audio_source, "rb")
                filename = Path(audio_source).name
            elif isinstance(audio_source, io.BytesIO):
                audio_source.name = filename
                file_obj = audio_source
            else:
                raise ValueError("audio_source должен быть путем к файлу или BytesIO")

            self.logger.debug(f"Отправка файла {filename} на транскрипцию...")

            resp = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=file_obj,
                **kwargs
            )

            text = resp.text if hasattr(resp, "text") else str(resp)
            self.logger.debug(f"Результат транскрипции: {text}")
            return text
        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API при транскрипции аудио: {e}")
            raise

class OpenAIImageModel(ImageGenerateModel):

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str = "gpt-image-1",
        default_n: int = 1,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=api_key)
        self.default_n = default_n
        self.logger.info(
            f"Image модель OpenAI инициализирована с моделью {self.model_name}, "
        )

    def generate(
        self,
        prompt: str,
        n: Optional[int] = None,
        **kwargs
    ) -> list[bytes]:

        try:
            resp = self.client.images.generate(
                model=self.model_name,
                prompt=prompt,
                n=n or self.default_n,
                **kwargs
            )

            images_data = []
            for i, img_info in enumerate(resp.data):
                img_bytes = base64.b64decode(img_info.b64_json)
                images_data.append(img_bytes)
                self.logger.debug(f"Изображение #{i+1} сгенерировано, размер {len(img_bytes)} байт")

            return images_data

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API при генерации изображения: {e}")
            raise
