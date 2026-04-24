"""Модуль настройки логирования для всего сервиса."""
import logging
import sys


def setup_logging():
    """Централизованная настройка формата и каналов вывода."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str):
    """Возвращает именованный логгер."""
    return logging.getLogger(name)