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

VoiceType = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
]

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
            # return resp.output_text
            return resp

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время generate(): {e}")
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
        
    def generate_with_shema_pydantic(
        self,
        input: list[dict[str, str]],
        text_format: Any,
        model_name: str = "gpt-5-nano",
        **kwargs
    ) -> ParsedResponse:
        # https://platform.openai.com/docs/guides/structured-outputs
        self._validate_token_limit(input)
        try:
            resp: ParsedResponse = self.client.responses.parse(
                model=model_name,
                input=input,
                text_format=text_format,
                **kwargs
            )
            self.logger.debug(f"Ответ модели с разбором в схему {text_format}: {resp.output_parsed}")
            # return resp.output_parsed
            return resp

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время generate_with_shema(): {e}")
            raise   
        
    def generate_with_schema(
        self,
        input: list[dict[str, str]],
        json_schema: dict,
        model_name: str = "gpt-5-nano",
        **kwargs
    ) -> Response:
        """
        Генерация текста с использованием ручной JSON Schema.

        :param input: список сообщений [{"role": "...", "content": "..."}]
        :param json_schema: схема в формате dict (см. https://platform.openai.com/docs/guides/structured-outputs)
        :param model_name: модель (по умолчанию gpt-5-nano)
        :return: Response с текстом, соответствующим схеме
        """
        self._validate_token_limit(input)
        try:
            resp: Response = self.client.responses.create(
                model=model_name,
                input=input,
                text=json_schema,
                **kwargs
            )
            self.logger.debug(
                f"Ответ модели с использованием json_schema={json_schema.get('format', {}).get('name')}: {resp.output_text}"
            )
            # return resp.output_text
            return resp

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время generate_with_shema(): {e}")
            raise        
    
    @staticmethod
    def complete_input(
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
       
    
class OpenAITTSModel(AudioTranscribeModel):
    """
    Класс для генерации аудио из текста с помощью OpenAI TTS (gpt-4o-mini-tts).
    """

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str = "gpt-4o-mini-tts",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"TTS модель OpenAI инициализирована с моделью {self.model_name}")

    def generate(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        voice: VoiceType = "alloy",
        response_format: str = "wav",
        as_bytes: bool = False,
        **kwargs
    ) -> Union[Path, bytes]:
        """
        Генерация речи из текста.

        :param text: Текст для озвучивания
        :param output_path: Путь для сохранения файла (если as_bytes=False)
        :param voice: Голос (по умолчанию alloy)
        :param response_format: Формат файла (wav/mp3/ogg)
        :param as_bytes: Если True — вернуть байты вместо пути к файлу
        :return: Path | bytes
        """
        try:
            self.logger.debug(f"Генерация речи из текста: '{text[:50]}...'")

            with self.client.audio.speech.with_streaming_response.create(
                model=self.model_name,
                voice=voice,
                input=text,
                response_format=response_format,
                **kwargs
            ) as response:
                if as_bytes:
                    audio_bytes = response.read()
                    self.logger.info("Аудио успешно сгенерировано (в байтах)")
                    return audio_bytes
                else:
                    if output_path is None:
                        output_path = Path(f"speech.{response_format}")
                    else:
                        output_path = Path(output_path)
                    response.stream_to_file(output_path)
                    self.logger.info(f"Аудио успешно сгенерировано: {output_path}")
                    return output_path

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API при генерации речи: {e}")
            raise
    

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
        output_path: Optional[Union[str, Path]] = None,
        as_bytes: bool = True,
        **kwargs
    ) -> Union[list[bytes], list[Path]]:
        """
        Сгенерировать изображения по текстовому описанию.

        :param prompt: описание изображения
        :param n: количество изображений
        :param output_path: путь для сохранения (если None — имя будет auto)
        :param as_bytes: если True — вернуть список байт, иначе список файлов
        :param kwargs: дополнительные параметры API (size, quality, и т.д.)
        :return: list[bytes] или list[Path]
        """
        try:
            resp = self.client.images.generate(
                model=self.model_name,
                prompt=prompt,
                n=n or self.default_n,
                **kwargs
            )

            results = []
            for i, img_info in enumerate(resp.data, start=1):
                img_bytes = base64.b64decode(img_info.b64_json)

                if as_bytes:
                    results.append(img_bytes)
                    self.logger.debug(
                        f"Изображение #{i} сгенерировано, размер {len(img_bytes)} байт"
                    )
                else:
                    if output_path is None:
                        file_path = Path(f"openai_image_{i}.png")
                    else:
                        output_path = Path(output_path)
                        if output_path.is_dir():
                            file_path = output_path / f"openai_image_{i}.png"
                        else:
                            stem = output_path.stem
                            file_path = output_path.with_name(f"{stem}_{i}{output_path.suffix}")

                    file_path.write_bytes(img_bytes)
                    results.append(file_path)
                    self.logger.info(f"Изображение #{i} сохранено: {file_path}")

            return results

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API при генерации изображения: {e}")
            raise
