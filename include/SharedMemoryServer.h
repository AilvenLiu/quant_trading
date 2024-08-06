#ifndef SHAREDMEMORYSERVER_H
#define SHAREDMEMORYSERVER_H

#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <sstream>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include "Logger.h"
#include "EClientSocket.h"
#include "EWrapper.h"

using namespace std;
using namespace boost::interprocess;

class SharedMemoryServer : public EWrapper {
private:
    EClientSocket *client;
    Logger *logger;
    shared_memory_object shm;
    mapped_region region;
    std::mutex dataMutex;
    stringstream dataBuffer;

    int nextOrderId;
    int requestId;

public:
    SharedMemoryServer(Logger *log);
    ~SharedMemoryServer();

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
    void writeToSharedMemory(const std::string &data);
};

#endif // SHAREDMEMORYSERVER_H