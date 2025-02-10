import logging

def configure_logger(name="imageSearch", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # 親ロガーへの伝播を無効化することで重複出力を防ぐ
    logger.propagate = False
    return logger