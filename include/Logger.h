#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>

enum LogLevel {
    INFO,
    WARNING,
    ERROR,
    DEBUG
};

class Logger {
private:
    std::ofstream logFile;
    std::mutex logMutex;
    LogLevel logLevel;

    std::string getLogLevelString(LogLevel level);

public:
    Logger(const std::string &filename, LogLevel level = INFO);
    ~Logger();

    void log(LogLevel level, const std::string &message);

    void info(const std::string &message);
    void warning(const std::string &message);
    void error(const std::string &message);
    void debug(const std::string &message);
};

#endif // LOGGER_H