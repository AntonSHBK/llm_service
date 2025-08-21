# Whisper + CTranslate2: Рейтинг по скорости (CPU)

| Ранг | Модель HF                                                                                                           | Размер / Квантование | Скорость на CPU | Точность       | Примечание                                               |
| ---- | ------------------------------------------------------------------------------------------------------------------- | -------------------- | --------------- | -------------- | -------------------------------------------------------- |
| 1    | [Systran/faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base)                                   | FP16 (CTranslate2)   | Очень быстрая   | Низкая/средняя | Подходит для работы в реальном времени, минимум ресурсов |
| 2    | [Zoont/faster-whisper-large-v3-turbo-int8-ct2](https://huggingface.co/Zoont/faster-whisper-large-v3-turbo-int8-ct2) | INT8 (CTranslate2)   | Быстрая         | Высокая        | Лучший баланс между скоростью и точностью                |
| 3    | [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium)                               | FP16                 | Быстрая         | Хорошая        | Средний размер, быстрее large, точнее base               |
| 4    | [jvh/whisper-large-v3-quant-ct2](https://huggingface.co/jvh/whisper-large-v3-quant-ct2)                             | INT8 (CTranslate2)   | Средняя         | Очень высокая  | Эффективное квантование large-v3                         |
| 5    | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3)                           | FP16                 | Медленная       | Максимальная   | Самая точная, но самая требовательная                    |
