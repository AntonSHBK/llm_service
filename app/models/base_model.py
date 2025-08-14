from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.services.token_manager import TokenManager
from app.utils.logging import get_logger
from app.settings import settings


class BaseLLMService(ABC):
    """
    Базовый абстрактный класс для работы с LLM.
    """

    def __init__(self, model_name: str, max_tokens: int = 1024, log_file: str | None = None):
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.logger = get_logger(self.__class__.__name__,
                                 log_dir=settings.LOG_DIR,
                                 log_file=log_file or f"{self.__class__.__name__.lower()}.log")

        self.token_manager = TokenManager(model_name)

        self.logger.info(f"Инициализация модели: {model_name}, max_tokens={max_tokens}")

    def count_tokens(self, text: str) -> int:
        """Подсчёт токенов для текста."""
        tokens = self.token_manager.count_tokens(text)
        self.logger.debug(f"Подсчёт токенов: {tokens} токенов для текста длиной {len(text)} символов")
        return tokens

    def check_token_limit(self, messages: list[dict[str, str]]) -> None:
        """Проверка, что суммарное количество токенов не превышает лимит."""
        total_tokens = sum(self.count_tokens(m["content"]) for m in messages)
        if total_tokens > self.max_tokens:
            self.logger.error(f"Превышен лимит токенов ({total_tokens}/{self.max_tokens})")
            raise ValueError(f"Превышен лимит токенов: {total_tokens} > {self.max_tokens}")
        self.logger.debug(f"Общее количество токенов: {total_tokens}")

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Отправка текстового запроса в модель."""
        pass

    @abstractmethod
    def generate_image(self, prompt: str, **kwargs) -> Any:
        """Генерация изображения по запросу."""
        pass

    @abstractmethod
    def transcribe_audio(self, file_path: Path, **kwargs) -> str:
        """Транскрипция аудиофайла."""
        pass
