#include "RealTimeData.h"

// Constructor
RealTimeData::RealTimeData(Logger *log) : client(nullptr), logger(log), nextOrderId(0), requestId(0), marketOpen(false) {
    // Initialize paths and open files for writing
    l1FilePath = "data/realtime_original_data/l1_data.csv";
    l2FilePath = "data/realtime_original_data/l2_data.csv";
    combinedFilePath = "data/daily_realtime_data/combined_data.csv";

    l1DataFile.open(l1FilePath, std::ios::app);
    l2DataFile.open(l2FilePath, std::ios::app);
    combinedDataFile.open(combinedFilePath, std::ios::app);

    connectToIB();
}

// Destructor
RealTimeData::~RealTimeData() {
    if (l1DataFile.is_open()) l1DataFile.close();
    if (l2DataFile.is_open()) l2DataFile.close();
    if (combinedDataFile.is_open()) combinedDataFile.close();
    if (client) delete client;
}

void RealTimeData::connectToIB() {
    client = new EClientSocket(this, nullptr);
    // Connection setup
    const char *host = "127.0.0.1";
    int port = 7496;
    int clientId = 0;

    if (client->eConnect(host, port, clientId)) {
        STX_LOGI(logger, "Connected to IB TWS");
    } else {
        STX_LOGE(logger, "Failed to connect to IB TWS");
    }
}

void RealTimeData::start() {
    while (true) {
        if (isMarketOpen()) {
            requestData();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else {
            std::this_thread::sleep_for(std::chrono::minutes(1));
        }
    }
}

bool RealTimeData::isMarketOpen() {
    // Use New York time zone
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    // Market open from 9:30 to 16:00
    return (tm.tm_hour >= 9 && tm.tm_hour <= 16) && !(tm.tm_hour == 9 && tm.tm_min < 30);
}

void RealTimeData::requestData() {
    Contract contract;
    contract.symbol = "SPY";
    contract.secType = "STK";
    contract.exchange = "SMART";
    contract.currency = "USD";

    // Request L1 and L2 data
    client->reqMktData(++requestId, contract, "", false, false, TagValueListSPtr());
    client->reqMktDepth(++requestId, contract, 5, true, TagValueListSPtr());
}

void RealTimeData::tickPrice(TickerId tickerId, TickType field, double price, const TickAttrib &attrib) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Tick Price: " << tickerId << " Field: " << field << " Price: " << price;
    STX_LOGI(logger, oss.str());

    // Write L1 data
    if (field == LAST || field == BID || field == ASK) {
        writeL1Data(oss.str());
    }
}

void RealTimeData::tickSize(TickerId tickerId, TickType field, Decimal size) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Tick Size: " << tickerId << " Field: " << field << " Size: " << size;
    STX_LOGI(logger, oss.str());

    // Write L1 data
    if (field == LAST_SIZE || field == BID_SIZE || field == ASK_SIZE) {
        writeL1Data(oss.str());
    }
}

void RealTimeData::updateMktDepth(TickerId id, int position, int operation, int side, double price, Decimal size) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Update Mkt Depth: " << id << " Position: " << position << " Operation: " << operation << " Side: " << side << " Price: " << price << " Size: " << size;
    STX_LOGI(logger, oss.str());
    

    // Write L2 data
    writeL2Data(oss.str());
}

void RealTimeData::updateMktDepthL2(TickerId id, int position, const std::string &marketMaker, int operation, int side, double price, Decimal size, bool isSmartDepth) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Update Mkt Depth L2: " << id << " Position: " << position << " MarketMaker: " << marketMaker << " Operation: " << operation << " Side: " << side << " Price: " << price << " Size: " << size << " SmartDepth: " << isSmartDepth;
    STX_LOGI(logger, oss.str());

    // Write L2 data
    writeL2Data(oss.str());
}

void RealTimeData::error(int id, int errorCode, const std::string &errorString, const std::string &advancedOrderRejectJson) {
    std::lock_guard<std::mutex> lock(dataMutex);

    std::ostringstream oss;
    oss << "Error ID: " << id << " Code: " << errorCode << " Msg: " << errorString;
    STX_LOGE(logger, oss.str());
}

void RealTimeData::nextValidId(OrderId orderId) {
    nextOrderId = orderId;
    STX_LOGI(logger, "Next valid order ID: " + std::to_string(orderId));
}

void RealTimeData::writeL1Data(const std::string &data) {
    if (l1DataFile.is_open()) {
        l1DataFile << data << std::endl;
    }
}

void RealTimeData::writeL2Data(const std::string &data) {
    if (l2DataFile.is_open()) {
        l2DataFile << data << std::endl;
    }
}

void RealTimeData::writeCombinedData(const std::string &data) {
    if (combinedDataFile.is_open()) {
        combinedDataFile << data << std::endl;
    }
}

std::string RealTimeData::calculateFeatures(const std::string &l1Data, const std::vector<std::string> &l2Data) {
    // Implement feature calculation based on L1 and L2 data
    // This should include combining relevant data and producing the desired output format
    return "calculated_features"; // Placeholder
}