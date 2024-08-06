#include "SharedMemoryServer.h"

// Constructor
SharedMemoryServer::SharedMemoryServer(Logger *log) : client(nullptr), logger(log), nextOrderId(0), requestId(0) {
    // Create shared memory object
    shm = shared_memory_object(create_only, "RealTimeData", read_write);
    shm.truncate(1024);  // Adjust size as needed
    region = mapped_region(shm, read_write);

    connectToIB();
}

// Destructor
SharedMemoryServer::~SharedMemoryServer() {
    if (client) delete client;
    shared_memory_object::remove("RealTimeData");
}

void SharedMemoryServer::connectToIB() {
    client = new EClientSocket(this, nullptr);
    // Connection setup
    const char *host = "127.0.0.1";
    int port = 7496;
    int clientId = 0;

    if (client->eConnect(host, port, clientId)) {
        logger->log("Connected to IB TWS");
    } else {
        logger->log("Failed to connect to IB TWS");
    }
}

void SharedMemoryServer::start() {
    while (true) {
        requestData();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void SharedMemoryServer::requestData() {
    Contract contract;
    contract.symbol = "SPY";
    contract.secType = "STK";
    contract.exchange = "SMART";
    contract.currency = "USD";

    // Request L1 and L2 data
    client->reqMktData(++requestId, contract, "", false, false, TagValueListSPtr());
    client->reqMktDepth(++requestId, contract, 5, true);
}

void SharedMemoryServer::tickPrice(TickerId tickerId, TickType field, double price, const TickAttrib &attrib) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Tick Price: " << tickerId << " Field: " << field << " Price: " << price;
    logger->log(oss.str());

    // Write to shared memory
    writeToSharedMemory(oss.str());
}

void SharedMemoryServer::tickSize(TickerId tickerId, TickType field, Decimal size) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Tick Size: " << tickerId << " Field: " << field << " Size: " << size;
    logger->log(oss.str());

    // Write to shared memory
    writeToSharedMemory(oss.str());
}

void SharedMemoryServer::updateMktDepth(TickerId id, int position, int operation, int side, double price, Decimal size) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Update Mkt Depth: " << id << " Position: " << position << " Operation: " << operation << " Side: " << side << " Price: " << price << " Size: " << size;
    logger->log(oss.str());

    // Write to shared memory
    writeToSharedMemory(oss.str());
}

void SharedMemoryServer::updateMktDepthL2(TickerId id, int position, const std::string &marketMaker, int operation, int side, double price, Decimal size, bool isSmartDepth) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Update Mkt Depth L2: " << id << " Position: " << position << " MarketMaker: " << marketMaker << " Operation: " << operation << " Side: " << side << " Price: " << price << " Size: " << size << " SmartDepth: " << isSmartDepth;
    logger->log(oss.str());

    // Write to shared memory
    writeToSharedMemory(oss.str());
}

void SharedMemoryServer::error(int id, int errorCode, const std::string &errorString, const std::string &advancedOrderRejectJson) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Error ID: " << id << " Code: " << errorCode << " Msg: " << errorString;
    logger->log(oss.str());
}

void SharedMemoryServer::nextValidId(OrderId orderId) {
    nextOrderId = orderId;
    logger->log("Next valid order ID: " + std::to_string(orderId));
}

void SharedMemoryServer::writeToSharedMemory(const std::string &data) {
    std::lock_guard<std::mutex> lock(dataMutex);

    // Copy data into shared memory
    std::memcpy(region.get_address(), data.c_str(), data.size());
}