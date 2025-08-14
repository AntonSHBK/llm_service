from pathlib import Path
from typing import Any, Generator, Optional

from openai import OpenAI
from openai import APIError

from app.models.base_model import BaseLLMService


class OpenAIService(BaseLLMService):
    def __init__(
        self,
        api_key: Optional[str],
        model_name: str = "gpt-4.1-nano",
        max_tokens: Optional[int] = 1024,
        **kwargs
    ):
        super().__init__(model_name, max_tokens=max_tokens, **kwargs)
        self.client = OpenAI(api_key=api_key)
        self.logger.info(f"Клиент OpenAI инициализирован с моделью {self.model_name}")

    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> str | Generator[str, None, None]:
        self.check_token_limit(messages)
        
        try:
            if stream:
                self.logger.debug("Starting streaming chat completion...")
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

            else:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs
                )
                content = resp.choices[0].message.content
                self.logger.debug(f"Full response: {content}")
                return content

        except APIError as e:
            self.logger.error(f"OpenAI API error during chat: {e}")
            raise

    def generate_image(self, prompt: str, **kwargs) -> Any:
        """
        Генерация изображения через универсальный Responses API.
        """
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[{
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]
                }],
                **kwargs
            )
            self.logger.debug("Image generation response received")
            return response
        except APIError as e:
            self.logger.error(f"OpenAI API error during image generation: {e}")
            raise

    def transcribe_audio(self, file_path: Path, **kwargs) -> str:
        """
        Транскрипция аудио — с помощью Responses API в случае мультимодальности,
        или специализированного endpoint, если доступен.
        """
        try:
            with open(file_path, "rb") as audio_file:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[{
                        "role": "user",
                        "content": [{"type": "input_audio", "audio_bytes": audio_file.read()}]
                    }],
                    **kwargs
                )
            text = response.output_text if hasattr(response, "output_text") else response.choices[0].message.content
            self.logger.debug(f"Transcription result: {text}")
            return text
        except APIError as e:
            self.logger.error(f"OpenAI API error during audio transcription: {e}")
            raise
