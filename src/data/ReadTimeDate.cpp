#include "RealTimeData.h"

RealTimeData::RealTimeData(Logger *log) : logger(log), client(new EClientSocket(this, nullptr)), marketOpen(false) {
    // Create file paths
    time_t t = time(0);  // get time now
    tm *now = localtime(&t);
    char dateStr[80];
    strftime(dateStr, sizeof(dateStr), "%Y-%m-%d", now);

    l1FilePath = "data/realtime_original_data/l1_" + string(dateStr) + ".csv";
    l2FilePath = "data/realtime_original_data/l2_" + string(dateStr) + ".csv";
    combinedFilePath = "data/daily_realtime_data/combined_" + string(dateStr) + ".csv";

    // Open file streams
    l1DataFile.open(l1FilePath, ios::app);
    l2DataFile.open(l2FilePath, ios::app);
    combinedDataFile.open(combinedFilePath, ios::app);

    if (!l1DataFile.is_open() || !l2DataFile.is_open() || !combinedDataFile.is_open()) {
        logger->error("Error opening files for writing");
    }
}

RealTimeData::~RealTimeData() {
    if (client->isConnected()) {
        client->eDisconnect();
    }
    delete client;

    if (l1DataFile.is_open()) {
        l1DataFile.close();
    }
    if (l2DataFile.is_open()) {
        l2DataFile.close();
    }
    if (combinedDataFile.is_open()) {
        combinedDataFile.close();
    }
}

void RealTimeData::connectToIB() {
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

bool RealTimeData::isMarketOpen() {
    using namespace std::chrono;
    system_clock::time_point now = system_clock::now();
    time_t tt = system_clock::to_time_t(now);
    tm local_tm = *localtime(&tt);

    // Set market open and close times in New York (EST)
    int openHour = 9;
    int openMinute = 30;
    int closeHour = 16;
    int closeMinute = 0;

    // Convert local time to EST (New York Time)
    tm est_tm = *localtime(&tt);
    est_tm.tm_hour = local_tm.tm_hour - 5;  // Adjust for EST (UTC-5)
    mktime(&est_tm);

    return (est_tm.tm_hour > openHour || (est_tm.tm_hour == openHour && est_tm.tm_min >= openMinute)) &&
           (est_tm.tm_hour < closeHour || (est_tm.tm_hour == closeHour && est_tm.tm_min < closeMinute));
}

void RealTimeData::requestData() {
    Contract contract;
    contract.symbol = "SPY";
    contract.secType = "STK";
    contract.exchange = "ARCA";  // Use ARCA or SMART for correct exchange
    contract.currency = "USD";

    client->reqMktData(requestId++, contract, "", false, false, TagValueListSPtr());
    client->reqMktDepth(requestId++, contract, 10, false, TagValueListSPtr());
}

void RealTimeData::start() {
    connectToIB();

    while (true) {
        if (isMarketOpen()) {
            if (!marketOpen) {
                logger->info("Market opened");
                marketOpen = true;
                requestData();
            }
        } else {
            if (marketOpen) {
                logger->info("Market closed");
                marketOpen = false;
                std::this_thread::sleep_for(std::chrono::seconds(3600));  // Sleep for 1 hour if market closed
                continue;
            }
            std::this_thread::sleep_for(std::chrono::seconds(60));  // Check every minute
            continue;
        }

        client->checkMessages();
        std::this_thread::sleep_for(std::chrono::seconds(1));  // Ensure a fixed interval of 1 second
    }
}

void RealTimeData::writeL1Data(const std::string &data) {
    lock_guard<mutex> lock(dataMutex);
    if (l1DataFile.is_open()) {
        l1DataFile << data << endl;
    }
}

void RealTimeData::writeL2Data(const std::string &data) {
    lock_guard<mutex> lock(dataMutex);
    if (l2DataFile.is_open()) {
        l2DataFile << data << endl;
    }
}

void RealTimeData::writeCombinedData(const std::string &data) {
    lock_guard<mutex> lock(dataMutex);
    if (combinedDataFile.is_open()) {
        combinedDataFile << data << endl;
    }
}

string RealTimeData::calculateFeatures(const string &l1Data, const vector<string> &l2Data) {
    // Perform feature engineering
    // Example: Calculate spread and total depth

    double bid = 0.0, ask = 0.0, last = 0.0, open = 0.0, high = 0.0, low = 0.0, close = 0.0, volume = 0.0;
    stringstream ss(l1Data);
    string token;
    int index = 0;

    while (getline(ss, token, ',')) {
        switch (index) {
            case 1: bid = stod(token); break;
            case 2: ask = stod(token); break;
            case 3: last = stod(token); break;
            case 4: open = stod(token); break;
            case 5: high = stod(token); break;
            case 6: low = stod(token); break;
            case 7: close = stod(token); break;
            case 8: volume = stod(token); break;
            default: break;
        }
        index++;
    }

    double spread = ask - bid;
    double totalDepth = 0.0;
    for (const auto &l2 : l2Data) {
        stringstream l2ss(l2);
        string l2Token;
        int l2Index = 0;
        while (getline(l2ss, l2Token, ',')) {
            if (l2Index == 6) {
                totalDepth += stod(l2Token); // Add size
            }
            l2Index++;
        }
    }

    // Create combined data with features
    stringstream featureData;
    featureData << l1Data << "," << spread << "," << totalDepth;
    return featureData.str();
}

void RealTimeData::tickPrice(TickerId tickerId, TickType field, double price, const TickAttrib &attrib) {
    if (field == BID || field == ASK || field == LAST || field == CLOSE || field == OPEN || field == HIGH || field == LOW) {
        time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        char buf[80];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

        stringstream ss;
        ss << buf << "," << tickerId << "," << field << "," << price;
        writeL1Data(ss.str());
        logger->info("L1 Data: " + ss.str());
    }
}

void RealTimeData::tickSize(TickerId tickerId, TickType field, int size) {
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    stringstream ss;
    ss << buf << "," << tickerId << "," << field << "," << size;
    writeL1Data(ss.str());
    logger->info("L1 Size: " + ss.str());
}

void RealTimeData::updateMktDepth(TickerId id, int position, int operation, int side, double price, int size) {
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    stringstream ss;
    ss << buf << "," << id << "," << position << "," << operation << "," << side << "," << price << "," << size;
    writeL2Data(ss.str());
    logger->info("L2 Data: " + ss.str());

    // Update combined data with feature engineering
    vector<string> l2Data = {ss.str()}; // Collect relevant L2 data for feature calculation
    string combinedData = calculateFeatures(ss.str(), l2Data);
    writeCombinedData(combinedData);
    logger->info("Combined Data: " + combinedData);
}

void RealTimeData::updateMktDepthL2(TickerId id, int position, const std::string &marketMaker, int operation, int side, double price, int size, bool isSmartDepth) {
    time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    stringstream ss;
    ss << buf << "," << id << "," << position << "," << marketMaker << "," << operation << "," << side << "," << price << "," << size << "," << isSmartDepth;
    writeL2Data(ss.str());
    logger->info("L2 Data L2: " + ss.str());

    // Update combined data with feature engineering
    vector<string> l2Data = {ss.str()}; // Collect relevant L2 data for feature calculation
    string combinedData = calculateFeatures(ss.str(), l2Data);
    writeCombinedData(combinedData);
    logger->info("Combined Data: " + combinedData);
}

void RealTimeData::error(int id, int errorCode, const std::string &errorString) {
    logger->error("Error: " + to_string(id) + ", Code: " + to_string(errorCode) + ", Msg: " + errorString);
}

void RealTimeData::nextValidId(OrderId orderId) {
    nextOrderId = orderId;
    logger->info("Next Valid Order ID: " + to_string(orderId));
}