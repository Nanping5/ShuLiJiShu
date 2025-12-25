"""
日志工具 - 统一的日志管理
"""
import logging
import sys
from pathlib import Path
from config import LogConfig


def setup_logger(name: str, log_file: Path = None, level: str = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别（可选）
    
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    log_level = getattr(logging, level or LogConfig.LEVEL)
    logger.setLevel(log_level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        fmt=LogConfig.FORMAT,
        datefmt=LogConfig.DATE_FORMAT
    )
    
    # 控制台处理器
    if LogConfig.CONSOLE_OUTPUT:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file is None:
        log_file = LogConfig.LOG_FILE
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器（简化接口）
    
    Args:
        name: 日志记录器名称
    
    Returns:
        日志记录器
    """
    return setup_logger(name)


class LoggerContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: logging.Logger, message: str):
        self.logger = logger
        self.message = message
    
    def __enter__(self):
        self.logger.info(f"开始: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"完成: {self.message}")
        else:
            self.logger.error(f"失败: {self.message} - {exc_val}")
        return False


if __name__ == "__main__":
    # 测试日志
    logger = get_logger("test")
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    
    # 测试上下文管理器
    with LoggerContext(logger, "测试任务"):
        logger.info("执行任务中...")
