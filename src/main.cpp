#include "Logger.h"
#include "SharedMemoryServer.h"

int main() {
    Logger logger("logs/realtime_data.log");
    SharedMemoryServer server(&logger);

    server.start();

    return 0;
}