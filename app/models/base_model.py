from abc import ABC, abstractmethod

from app.utils.logging import get_logger
from app.settings import settings


class BaseLLMService(ABC):

    def __init__(
        self, 
        model_name: str,
        log_file: str | None = None
    ):
        self.model_name = model_name

        self.logger = get_logger(
            self.__class__.__name__,
            log_dir=settings.LOG_DIR,
            log_file=log_file or f"{self.__class__.__name__.lower()}.log"
        )        

        self.logger.info(f"Инициализация модели: {model_name}, ")