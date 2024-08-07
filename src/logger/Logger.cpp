#include "Logger.h"

Logger::Logger(const std::string& filename, LogLevel level)
    : logLevel(level) {
    logFile.open(filename, std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << filename << std::endl;
    }
}

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

std::string Logger::getTimestamp() const {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level <= logLevel) {
        std::lock_guard<std::mutex> lock(logMutex);
        logFile << "[" << getTimestamp() << "] "
                << "[" << logLevelToString(level) << "] "
                << message << std::endl;
    }
}

std::string Logger::logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::FATAL: return "FATAL";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::INFO: return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        default: return "UNKNOWN";
    }
}