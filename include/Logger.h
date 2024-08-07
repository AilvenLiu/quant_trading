#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <ctime>

// 定义日志级别的枚举
enum class LogLevel {
    FATAL,
    ERROR,
    WARNING,
    INFO,
    DEBUG
};

class Logger {
private:
    std::ofstream logFile;
    std::mutex logMutex;
    LogLevel logLevel;

    std::string getTimestamp() const;

public:
    Logger(const std::string& filename, LogLevel level = LogLevel::INFO);
    ~Logger();

    void log(LogLevel level, const std::string& message);

    static std::string logLevelToString(LogLevel level);
};

// 定义日志宏
#define STX_LOGF(logger, message) (logger)->log(LogLevel::FATAL, message)
#define STX_LOGE(logger, message) (logger)->log(LogLevel::ERROR, message)
#define STX_LOGW(logger, message) (logger)->log(LogLevel::WARNING, message)
#define STX_LOGI(logger, message) (logger)->log(LogLevel::INFO, message)
#define STX_LOGD(logger, message) (logger)->log(LogLevel::DEBUG, message)

#endif // LOGGER_H