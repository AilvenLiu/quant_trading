#include "SharedMemoryServer.h"

SharedMemoryServer::SharedMemoryServer(Logger *log) : logger(log), client(new EClientSocket(this, nullptr)) {
    shm = shared_memory_object(create_only, "SharedMemory", read_write);
    shm.truncate(65536); // Set the size of the shared memory segment
    region = mapped_region(shm, read_write);

    std::memset(region.get_address(), 0, region.get_size());

    nextOrderId = 0;
    requestId = 1; // Start with 1
}

SharedMemoryServer::~SharedMemoryServer() {
    delete client;
    shared_memory_object::remove("SharedMemory");
}

void SharedMemoryServer::connectToIB() {
    const char *host = "";
    int port = 7496;
    int clientId = 0;

    while (!client->isConnected()) {
        bool isConnected = client->eConnect(host, port, clientId);
        if (isConnected) {
            logger->info("Connected to TWS server");
            client->setServerLogLevel(5);
        } else {
            logger->error("Could not connect to TWS server, retrying in 5 seconds...");
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
}

void SharedMemoryServer::requestData() {
    Contract contract;
    contract.symbol = "SPY";
    contract.secType = "STK";
    contract.exchange = "ARCA";
    contract.currency = "USD";

    client->reqMktData(requestId++, contract, "", false, false, TagValueListSPtr());
    client->reqMktDepth(requestId++, contract, 10, false, TagValueListSPtr());
}

void SharedMemoryServer::start() {
    connectToIB();
    requestData();

    while (client->isConnected()) {
        client->checkMessages();
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Ensure a fixed interval of 1 second
    }
}

void SharedMemoryServer::writeToSharedMemory(const std::string &data) {
    std::lock_guard<std::mutex> lock(dataMutex);
    std::memset(region.get_address(), 0, region.get_size());
    std::memcpy(region.get_address(), data.c_str(), std::min(data.size(), region.get_size() - 1));
}

void SharedMemoryServer::tickPrice(TickerId tickerId, TickType field, double price, const TickAttrib &attrib) {
    if (field == BID || field == ASK || field == LAST || field == CLOSE || field == OPEN || field == HIGH || field == LOW) {
        time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        char buf[80];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

        stringstream ss;
        ss << buf << "," << tickerId << "," << field << "," << price;

        logger->info("L1 Data: " + ss.str());

        writeToSharedMemory(ss.str());
    }
}

void SharedMemoryServer::tickSize(TickerId tickerId, TickType field, int size) {
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    stringstream ss;
    ss << buf << "," << tickerId << "," << field << "," << size;

    logger->info("L1 Size: " + ss.str());

    writeToSharedMemory(ss.str());
}

void SharedMemoryServer::updateMktDepth(TickerId id, int position, int operation, int side, double price, int size) {
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    stringstream ss;
    ss << buf << "," << id << "," << position << "," << operation << "," << side << "," << price << "," << size;

    logger->info("L2 Data: " + ss.str());

    writeToSharedMemory(ss.str());
}

void SharedMemoryServer::updateMktDepthL2(TickerId id, int position, const std::string &marketMaker, int operation, int side, double price, int size, bool isSmartDepth) {
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    stringstream ss;
    ss << buf << "," << id << "," << position << "," << marketMaker << "," << operation << "," << side << "," << price << "," << size << "," << isSmartDepth;

    logger->info("L2 Data L2: " + ss.str());

    writeToSharedMemory(ss.str());
}

void SharedMemoryServer::error(int id, int errorCode, const std::string &errorString) {
    logger->error("Error: " + to_string(id) + ", Code: " + to_string(errorCode) + ", Msg: " + errorString);
}

void SharedMemoryServer::nextValidId(OrderId orderId) {
    nextOrderId = orderId;
    logger->info("Next Valid Order ID: " + to_string(orderId));
}