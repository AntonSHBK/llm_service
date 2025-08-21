import base64
import io
from pathlib import Path
from typing import Generator, Optional, Any, Union, BinaryIO, IO

from openai import OpenAI, APIError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai._streaming import Stream

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
        model_name: str = "gpt-4.1-nano",
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
        # self.logger.debug(f"Токены в норме: {total_tokens}/{self.max_tokens}")

    def generate(self, messages: list[dict[str, str]], **kwargs) -> str:
        # https://platform.openai.com/docs/api-reference/chat/create
        self._validate_token_limit(messages)

        try:
            resp: ChatCompletion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            content: str = resp.choices[0].message.content or ""
            self.logger.debug(f"Ответ модели: {content}")
            return content

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время generate_text: {e}")
            raise

    def generate_stream(
        self, messages: list[dict[str, str]], **kwargs
    ) -> Generator[str, None, None]:
        # https://platform.openai.com/docs/api-reference/chat/create
        self._validate_token_limit(messages)

        try:
            stream_resp: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs
            )

            def stream_generator() -> Generator[str, None, None]:
                for chunk in stream_resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        self.logger.debug(f"Stream chunk: {delta}")
                        yield delta

            return stream_generator()

        except APIError as e:
            self.logger.error(f"Ошибка OpenAI API во время generate_text_stream: {e}")
            raise

class OpenAIAudioTranscribeModel(AudioTranscribeModel):

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
