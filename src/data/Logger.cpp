#include "Logger.h"

Logger::Logger(const std::string &filename, LogLevel level) : logLevel(level) {
    logFile.open(filename, std::ios::out | std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error opening log file: " << filename << std::endl;
    }
}

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

std::string Logger::getLogLevelString(LogLevel level) {
    switch (level) {
        case INFO:
            return "INFO";
        case WARNING:
            return "WARNING";
        case ERROR:
            return "ERROR";
        case DEBUG:
            return "DEBUG";
        default:
            return "UNKNOWN";
    }
}

void Logger::log(LogLevel level, const std::string &message) {
    if (level >= logLevel) {
        std::lock_guard<std::mutex> lock(logMutex);
        logFile << "[" << getLogLevelString(level) << "] " << message << std::endl;
    }
}

void Logger::info(const std::string &message) {
    log(INFO, message);
}

void Logger::warning(const std::string &message) {
    log(WARNING, message);
}

void Logger::error(const std::string &message) {
    log(ERROR, message);
}

void Logger::debug(const std::string &message) {
    log(DEBUG, message);
}