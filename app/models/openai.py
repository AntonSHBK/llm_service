import base64
import io
from pathlib import Path
from typing import Generator, Optional, Any, Union, BinaryIO, IO

from openai import OpenAI, APIError

from app.models.base_model import BaseLLMService
from app.services.token_manager import TokenManager


class OpenAIChatModel(BaseLLMService):
    """
    Сервис для работы с чат-моделями OpenAI (обычный и потоковый режим).
    """

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str = "gpt-4.1-nano",
        max_tokens: Optional[int] = 4048,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"Chat модель OpenAI инициализирован с моделью {self.model_name}")

    def _validate_token_limit(self, messages: list[dict[str, str]]) -> None:
        """
        Проверка лимита токенов для сообщений.
        Если max_tokens=None — проверка пропускается.
        """
        if self.max_tokens is None:
            self.logger.debug("Лимит токенов не установлен — проверка пропущена.")
            return
        total_tokens = TokenManager.count_message_tokens(messages, self.model_name)
        if total_tokens > self.max_tokens:
            self.logger.warning(f"Превышен лимит токенов ({total_tokens}/{self.max_tokens})")
            raise ValueError(f"Превышен лимит токенов: {total_tokens}/{self.max_tokens}")
        self.logger.debug(f"Токены в норме: {total_tokens}/{self.max_tokens}")

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        # https://platform.openai.com/docs/api-reference/responses/create
        """
        Обычный чат-запрос — возвращает полный ответ модели.
        """
        self._validate_token_limit(messages)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            content = resp.choices[0].message.content
            self.logger.debug(f"Ответ модели: {content}")
            return content
        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время обычного чата: {e}")
            raise

    def chat_stream(self, messages: list[dict[str, str]], **kwargs) -> Generator[str, None, None]:
        # https://platform.openai.com/docs/api-reference/responses/create
        """
        Потоковый чат-запрос — возвращает ответ частями.
        """
        self._validate_token_limit(messages)
        try:
            stream_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs
            )

            def stream_generator():
                for chunk in stream_resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        self.logger.debug(f"Stream chunk: {delta}")
                        yield delta

            return stream_generator()
        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время потокового чата: {e}")
            raise


class OpenAIAudioModel(BaseLLMService):
    """
    Модель для транскрипции аудио с помощью OpenAI.
    """

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str = "gpt-4o-mini-transcribe",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"Audio модель OpenAI инициализирована с моделью {self.model_name}")
      
    def transcribe(
        self,
        audio_source: Union[str, Path, io.BytesIO],
        filename: Optional[str] = None,
        language: str = "ru",
        **kwargs
    ) -> str:
        """
        Транскрипция аудио.
        :param audio_source: путь к файлу или файловый поток BytesIO
        :param filename: имя файла (обязательно при BytesIO)
        :param language: язык аудио (ISO-639-1), по умолчанию "ru"
        """
        try:
            if isinstance(audio_source, (str, Path)):
                file_obj = open(audio_source, "rb")
                fname = Path(audio_source).name
            elif isinstance(audio_source, io.BytesIO):
                if not filename:
                    filename = "audio.wav"
                audio_source.name = filename
                fname = filename
                file_obj = audio_source
            else:
                raise ValueError("audio_source должен быть путем к файлу или BytesIO")

            self.logger.debug(f"Отправка файла {fname} на транскрипцию...")

            resp = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=file_obj,
                language=language,
                **kwargs
            )

            text = resp.text if hasattr(resp, "text") else str(resp)
            self.logger.debug(f"Результат транскрипции: {text}")
            return text
        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API при транскрипции аудио: {e}")
            raise

class OpenAIImageModel(BaseLLMService):
    """
    Модель для генерации изображений с помощью OpenAI.
    """

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
        """
        Генерация изображения по текстовому описанию.
        Возвращает список байтов (decoded PNG/JPEG/WebP).
        """
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
