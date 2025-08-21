import io
import json
import requests
from pathlib import Path
from typing import Optional, Union, Generator, Optional

from yandex_cloud_ml_sdk import YCloudML
from yandex_cloud_ml_sdk.exceptions import YCloudMLError

from app.models.base_model import TextGanerateModel, ImageGenerateModel, AudioTranscribeModel
from app.services.token_manager import TokenManager


class YandexChatModel(TextGanerateModel):
    # https://yandex.cloud/ru/docs/foundation-models/operations/yandexgpt/create-chat
    def __init__(
        self,
        folder_id: str,
        auth: Optional[str],
        model_name: str = "yandexgpt",
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.max_tokens = max_tokens
        self.sdk = YCloudML(folder_id=folder_id, auth=auth)
        self.model = self.sdk.models.completions(model_name)
        self.logger.info(f"Yandex Cloud Chat модель инициализирована: {model_name}")

    def _validate_token_limit(self, messages: list[dict[str, str]]) -> None:
        if self.max_tokens is None:
            self.logger.debug("Лимит токенов не установлен — пропускаем проверку.")
            return
        total_tokens = TokenManager.count_message_tokens(messages, self.model_name)
        if total_tokens > self.max_tokens:
            self.logger.warning(f"Токенов больше лимита ({total_tokens}/{self.max_tokens})")
            raise ValueError("Превышен лимит токенов")
        self.logger.debug(f"Токены в норме: {total_tokens}/{self.max_tokens}")

    def generate(self, messages: list[dict[str, str]], **kwargs) -> 'GPTModelResult':
        """
        Генерация ответа (синхронно) в формате списка сообщений.
        """
        self._validate_token_limit(messages)
        try:
            response: 'GPTModelResult' = self.model.configure(**kwargs).run(messages)
            self.logger.debug(f"Yandex Cloud response: {response}")
            return response.alternatives[0]
        except YCloudMLError as e:
            self.logger.error(f"Yandex Cloud ML error in generate_text: {e}")
            raise

    def generate_stream(self, messages: list[dict[str, str]], **kwargs) -> Generator[str, None, None]:
        return self.generate(messages, **kwargs)


class YandexArtModel(ImageGenerateModel):
    # https://yandex.cloud/ru/docs/foundation-models/operations/yandexart/request
    def __init__(
        self,
        folder_id: str,
        auth: str,
        model_name: str = "yandex-art",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.sdk = YCloudML(folder_id=folder_id, auth=auth)
        self.model = self.sdk.models.image_generation(model_name)
        self.logger.info(f"YandexART модель инициализирована: {model_name}")

    def generate(
        self,
        prompt: Union[str, list[Union[str, dict]]],
        output_path: Optional[str] = None,
        **kwargs
    ):
        """
        Сгенерировать изображение по промту.
        :param prompt: строка или список строк/словари вида {"text": "...", "weight": N}
        :param output_path: путь для сохранения результата (если нужен файл)
        :param kwargs: width_ratio, height_ratio, seed и другие параметры
        :return: объект ImageGenerationModelResult
        """
        try:
            model_cfg = self.model.configure(**kwargs)
            operation = model_cfg.run_deferred(prompt)
            result = operation.wait()

            if output_path:
                path = Path(output_path)
                path.write_bytes(result.image_bytes)
                self.logger.info(f"Изображение сохранено в {output_path}")

            return result

        except YCloudMLError as e:
            self.logger.error(f"Ошибка YandexART: {e}")
            raise


class YandexSpeechToTextModel(AudioTranscribeModel):
    # https://yandex.cloud/ru/docs/speechkit/

    def __init__(
        self,
        folder_id: str,
        auth: str,
        model_name: str = "general",
        language: str = "ru-RU",
        **kwargs
    ):
        """
        :param folder_id: ID каталога Yandex Cloud
        :param auth: API-ключ сервисного аккаунта
        :param model_name: имя модели SpeechKit (обычно "general")
        :param language: язык распознавания ("ru-RU", "en-US", ...)
        """
        super().__init__(model_name, **kwargs)
        self.language = language
        self.sdk = YCloudML(folder_id=folder_id, auth=auth)
        self.model = self.sdk.models.speech_to_text(model_name)
        self.logger.info(f"Yandex SpeechKit модель инициализирована: {model_name}")

    def generate(
        self,
        audio_source: Union[str, Path, io.BytesIO],
        filename: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Транскрипция аудио.
        :param audio_source: путь к файлу или BytesIO
        :param filename: имя файла (нужно при BytesIO)
        :param kwargs: параметры SpeechKit (например, enable_automatic_punctuation=True)
        :return: текст транскрипции
        """
        try:
            if isinstance(audio_source, (str, Path)):
                with open(audio_source, "rb") as f:
                    file_bytes = f.read()
            elif isinstance(audio_source, io.BytesIO):
                file_bytes = audio_source.read()
            else:
                raise ValueError("audio_source должен быть путем к файлу или BytesIO")

            self.logger.debug("Отправка аудио в SpeechKit...")

            operation = self.model.run_deferred(
                file_bytes,
                language=self.language,
                **kwargs
            )
            result = operation.wait()

            text = result.text if hasattr(result, "text") else str(result)
            self.logger.info(f"Транскрипция завершена: {len(text)} символов")
            return text

        except YCloudMLError as e:
            self.logger.error(f"Ошибка Yandex SpeechKit: {e}")
            raise


class YandexSpeechToTextModel(AudioTranscribeModel):
    # https://yandex.cloud/ru/docs/speechkit/quickstart/stt-quickstart-v1

    def __init__(
        self,
        api_key: str,
        model_name: str = "speechkit-stt",
        language: str = "ru-RU",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.language = language
        self.api_url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"

    def generate(
        self,
        audio_source: Union[str, Path, io.BytesIO],
        filename: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Транскрибировать аудио (до 30 сек / <=1MB).
        :param audio_source: путь к файлу или BytesIO
        :param filename: имя файла (для BytesIO)
        :param kwargs: дополнительные параметры (lang и др.)
        :return: распознанный текст
        """
        if isinstance(audio_source, (str, Path)):
            with open(audio_source, "rb") as f:
                audio_bytes = f.read()
        elif isinstance(audio_source, io.BytesIO):
            audio_bytes = audio_source.read()
        else:
            raise ValueError("audio_source должен быть путем к файлу или BytesIO")

        lang = kwargs.get("lang", self.language)

        headers = {
            "Authorization": f"Api-Key {self.api_key}"
        }
        params = {
            "lang": lang
        }

        self.logger.debug(f"Отправка аудио в Yandex STT, lang={lang}, size={len(audio_bytes)} байт")

        response = requests.post(
            self.api_url,
            headers=headers,
            params=params,
            data=audio_bytes
        )

        if response.status_code != 200:
            self.logger.error(f"Ошибка STT API: {response.status_code}, {response.text}")
            raise RuntimeError(f"STT API error: {response.status_code}, {response.text}")

        result = response.json()
        self.logger.debug(f"STT ответ: {json.dumps(result, ensure_ascii=False)}")

        if "result" not in result:
            raise RuntimeError(f"Ошибка транскрипции: {result}")

        return result["result"]
