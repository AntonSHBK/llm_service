from functools import lru_cache
from typing import Optional

import tiktoken

from app.utils.logging import get_logger
from app.settings import settings


class TokenManager:
    """
    Класс для подсчёта токенов в тексте для разных моделей LLM.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = get_logger(self.__class__.__name__,
                                 log_dir=settings.LOG_DIR,
                                 log_file="token_manager.log")
        self.encoder = self._get_encoder(model_name)
        self.logger.debug(f"TokenManager инициализирован для модели: {model_name}")

    @staticmethod
    @lru_cache(maxsize=10)
    def _get_encoder(model_name: str):
        """
        Кешированное получение энкодера для модели.
        Если модель неизвестна — используем общий BPE.
        """
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            default_encoding = "cl100k_base"
            logger = get_logger("TokenManagerInit",
                                log_dir=settings.LOG_DIR,
                                log_file="token_manager.log")
            logger.warning(f"Не удалось найти энкодер для '{model_name}', используем {default_encoding}")
            return tiktoken.get_encoding(default_encoding)

    def count_tokens(self, text: str) -> int:
        """
        Подсчёт количества токенов в тексте.
        """
        if not text:
            return 0
        tokens = len(self.encoder.encode(text))
        self.logger.debug(f"Текст длиной {len(text)} символов → {tokens} токенов")
        return tokens

    def count_message_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Подсчёт токенов в списке сообщений (role + content).
        """
        total_tokens = 0
        for m in messages:
            total_tokens += self.count_tokens(m.get("content", ""))
            total_tokens += self.count_tokens(m.get("role", ""))
        self.logger.debug(f"Суммарно в сообщениях: {total_tokens} токенов")
        return total_tokens

    def check_limit(self, messages: list[dict[str, str]], max_tokens: Optional[int] = None) -> None:
        """
        Проверка лимита токенов. Бросает исключение, если превышен.
        """
        limit = max_tokens or settings.max_tokens
        total_tokens = self.count_message_tokens(messages)
        if total_tokens > limit:
            self.logger.error(f"Превышен лимит токенов: {total_tokens}/{limit}")
            raise ValueError(f"Превышен лимит токенов: {total_tokens}/{limit}")
        self.logger.debug(f"Токены в норме: {total_tokens}/{limit}")
