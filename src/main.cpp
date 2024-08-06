#include "SharedMemoryServer.h"
#include "Logger.h"

int main() {
    Logger logger("logs/realtime_data.log", INFO);
    SharedMemoryServer server(&logger);
    server.start();
    return 0;
}