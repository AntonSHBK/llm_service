# app/services/token_manager.py
import tiktoken
from functools import lru_cache
from app.utils.logging import get_logger
from app.settings import settings

class TokenManager:
    logger = get_logger("TokenManager", log_dir=settings.LOG_DIR)

    @classmethod
    @lru_cache(maxsize=5)
    def _get_encoder(cls, model_name: str):
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: str) -> int:
        if not text:
            return 0
        encoder = cls._get_encoder(model_name)
        tokens = len(encoder.encode(text))
        cls.logger.debug(
            f"Текст длиной {len(text)} символов → {tokens} токенов (модель='{model_name}')"
        )
        return tokens

    @classmethod
    def count_message_tokens(cls, messages: list[dict], model_name: str) -> int:
        total_tokens = 0
        for m in messages:
            total_tokens += cls.count_tokens(m.get("role", ""), model_name)
            total_tokens += cls.count_tokens(m.get("content", ""), model_name)
        cls.logger.debug(f"Общее количество токенов в сообщениях: {total_tokens}")
        return total_tokens
