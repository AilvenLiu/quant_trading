#ifndef REALTIMEDATA_H
#define REALTIMEDATA_H

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <chrono>
#include <thread>
#include <ctime>
#include <mutex>
#include <iomanip>
#include "EClientSocket.h"
#include "EWrapper.h"
#include "Logger.h"

using namespace std;

class RealTimeData : public EWrapper {
private:
    EClientSocket *client;
    Logger *logger;
    ofstream l1DataFile;
    ofstream l2DataFile;
    ofstream combinedDataFile;
    int nextOrderId;
    int requestId;
    string l1FilePath;
    string l2FilePath;
    string combinedFilePath;
    mutex dataMutex;
    bool marketOpen;

    // Helper functions for feature engineering
    string calculateFeatures(const string &l1Data, const vector<string> &l2Data);

public:
    RealTimeData(Logger *log);
    ~RealTimeData();

    // EWrapper interface implementations
    void tickPrice(TickerId tickerId, TickType field, double price, const TickAttrib &attrib) override;
    void tickSize(TickerId tickerId, TickType field, int size) override;
    void updateMktDepth(TickerId id, int position, int operation, int side, double price, int size) override;
    void updateMktDepthL2(TickerId id, int position, const std::string &marketMaker, int operation, int side, double price, int size, bool isSmartDepth) override;
    void error(int id, int errorCode, const std::string &errorString) override;
    void nextValidId(OrderId orderId) override;

    void start();

    // Helper functions
    void connectToIB();
    void requestData();
    bool isMarketOpen();
    void writeL1Data(const std::string &data);
    void writeL2Data(const std::string &data);
    void writeCombinedData(const std::string &data);
};

#endif // REALTIMEDATA_H