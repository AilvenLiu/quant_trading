/**************************************************************************
 * This file is part of the OpenSTX project.
 *
 * OpenSTX (Open Smart Trading eXpert) is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OpenSTX is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OpenSTX. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Ailven.LIU
 * Email: ailven.x.liu@gmail.com
 * Date: 2024
 *************************************************************************/

#include <numeric>
#include "RealTimeData.h"

// Constructor
RealTimeData::RealTimeData(const std::shared_ptr<Logger>& log) : client(nullptr), logger(log), nextOrderId(0), requestId(0), yesterdayClose(0.0) {

    auto now = std::chrono::system_clock::now();
    std::time_t in_time_t = std::chrono::system_clock::to_time_t(now);

    // Format the time into a string
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", std::localtime(&in_time_t));
    std::string date_today(buffer);

    
    // Initialize paths and open files for writing
    l1FilePath = "data/realtime_original_data/l1_data_" + date_today + ".csv";
    l2FilePath = "data/realtime_original_data/l2_data_" + date_today + ".csv";
    combinedFilePath = "data/daily_realtime_data/combined_data_" + date_today + ".csv";

    l1DataFile.open(l1FilePath, std::ios::app);
    l2DataFile.open(l2FilePath, std::ios::app);
    combinedDataFile.open(combinedFilePath, std::ios::app);

    // Write headers to the files
    l1DataFile << "Datetime,Bid,Ask,Last,Open,High,Low,Close,Volume\n";
    l2DataFile << "Datetime,Position,BidPrice,BidSize,AskPrice,AskSize\n";
    combinedDataFile << "Datetime,Open,High,Low,Close,Volume,BidAskSpread,Midpoint,PriceChange,L2Depth,Gap\n";

    connectToIB();
}

// Destructor
RealTimeData::~RealTimeData() {
    if (l1DataFile.is_open()) l1DataFile.close();
    if (l2DataFile.is_open()) l2DataFile.close();
    if (combinedDataFile.is_open()) l1DataFile.close();
    if (client) client.reset();
    boost::interprocess::shared_memory_object::remove("RealTimeData");
}

void RealTimeData::connectToIB() {
    client = std::make_shared<EClientSocket>(this, nullptr);
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
    // Initialize shared memory
    shm = boost::interprocess::shared_memory_object(boost::interprocess::create_only, "RealTimeData", boost::interprocess::read_write);
    shm.truncate(1024);  // Adjust size as needed
    region = boost::interprocess::mapped_region(shm, boost::interprocess::read_write);

    std::time_t lastMinute = std::time(nullptr) / 60;

    while (true) {
        if (isMarketOpen()) {
            requestData();

            std::time_t currentMinute = std::time(nullptr) / 60;
            if (currentMinute != lastMinute) {
                aggregateMinuteData();
                lastMinute = currentMinute;
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));  // Adjust frequency as needed
        } else {
            std::this_thread::sleep_for(std::chrono::minutes(1));  // Check again after 1 minute
        }
    }
}

bool RealTimeData::isMarketOpen() {
    std::time_t nyTime = getNYTime();
    std::tm *tm = std::localtime(&nyTime);

    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    STX_LOGI(logger, "Current New York Time: " + std::string(buf));

    // Market open from 9:30 to 16:00
    return (tm->tm_hour > 9 && tm->tm_hour < 16) || (tm->tm_hour == 9 && tm->tm_min >= 30);
}

std::time_t RealTimeData::getNYTime() {
    std::time_t t = std::time(nullptr);
    std::tm *utc_tm = std::gmtime(&t);
    utc_tm->tm_hour -= 4;  // UTC-4 for Eastern Daylight Time (EDT)
    return std::mktime(utc_tm);
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

    if (field == LAST) {
        l1Prices.push_back(price);
    }

    // Write to shared memory
    std::ostringstream oss;
    oss << "Tick Price: " << tickerId << " Field: " << field << " Price: " << price;
    STX_LOGI(logger, oss.str());
    writeToSharedMemory(oss.str());
}

void RealTimeData::tickSize(TickerId tickerId, TickType field, Decimal size) {
    std::lock_guard<std::mutex> lock(dataMutex);

    if (field == LAST_SIZE) {
        l1Volumes.push_back(size);
    }

    // Write to shared memory
    std::ostringstream oss;
    oss << "Tick Size: " << tickerId << " Field: " << field << " Size: " << size;
    STX_LOGI(logger, oss.str());
    writeToSharedMemory(oss.str());
}

void RealTimeData::updateMktDepth(TickerId id, int position, int operation, int side, double price, Decimal size) {
    std::lock_guard<std::mutex> lock(dataMutex);
    processL2Data(position, price, size, side);

    std::ostringstream oss;
    oss << "Update Mkt Depth: " << id << " Position: " << position << " Operation: " << operation << " Side: " << side << " Price: " << price << " Size: " << size;
    STX_LOGI(logger, oss.str());
    writeToSharedMemory(oss.str());
}

void RealTimeData::updateMktDepthL2(TickerId id, int position, const std::string &marketMaker, int operation, int side, double price, Decimal size, bool isSmartDepth) {
    std::lock_guard<std::mutex> lock(dataMutex);
    processL2Data(position, price, size, side);

    std::ostringstream oss;
    oss << "Update Mkt Depth L2: " << id << " Position: " << position << " MarketMaker: " << marketMaker << " Operation: " << operation << " Side: " << side << " Price: " << price << " Size: " << size << " SmartDepth: " << isSmartDepth;
    STX_LOGI(logger, oss.str());
    writeToSharedMemory(oss.str());
}

void RealTimeData::processL2Data(int position, double price, Decimal size, int side) {
    std::map<std::string, double> data;
    data["Position"] = position;
    data["Price"] = price;
    data["Size"] = size;
    data["Side"] = side;
    l2Data.push_back(data);

    std::time_t now = std::time(nullptr);
    std::ostringstream datetimeStream;
    datetimeStream << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S");
    std::string datetime = datetimeStream.str();

    std::ostringstream oss;
    oss << datetime << "," << position << ",";
    if (side == 1) {  // Bid
        oss << price << "," << size << ",,";
    } else if (side == 0) {  // Ask
        oss << "," << "," << price << "," << size;
    }
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

void RealTimeData::aggregateMinuteData() {
    if (l1Prices.empty()) {
        return;  // No data to aggregate
    }

    std::lock_guard<std::mutex> lock(dataMutex);

    double open = l1Prices.front();
    double close = l1Prices.back();
    double high = *std::max_element(l1Prices.begin(), l1Prices.end());
    double low = *std::min_element(l1Prices.begin(), l1Prices.end());
    double volume = std::accumulate(l1Volumes.begin(), l1Volumes.end(), 0.0);

    double bidAskSpread = 0.0, midpoint = 0.0, priceChange = 0.0, l2BidTotalDepth = 0.0, l2AskTotalDepth = 0.0;
    for (const auto &data : l2Data) {
        if (data.at("Side") == 1) {  // Bid side
            l2BidTotalDepth += data.at("Size");
        } else if (data.at("Side") == 0) {  // Ask side
            l2AskTotalDepth += data.at("Size");
        }
    }

    if (!l2Data.empty()) {
        bidAskSpread = l2Data.back().at("Price") - l2Data.front().at("Price");
        midpoint = (l2Data.front().at("Price") + l2Data.back().at("Price")) / 2.0;
    }

    double gap = open - yesterdayClose;
    yesterdayClose = close;  // Update for the next day

    std::stringstream combinedData;
    combinedData << getNYTime() << "," << open << "," << high << "," << low << "," << close << "," << volume << ","
                 << bidAskSpread << "," << midpoint << "," << priceChange << "," << l2BidTotalDepth << "," << l2AskTotalDepth << "," << gap;

    writeCombinedData(combinedData.str());

    // Clear temporary data
    l1Prices.clear();
    l1Volumes.clear();
    l2Data.clear();
}

void RealTimeData::writeCombinedData(const std::string &data) {
    if (combinedDataFile.is_open()) {
        combinedDataFile << data << std::endl;
    }
}

void RealTimeData::writeL2Data(const std::string &data) {
    if (l2DataFile.is_open()) {
        l2DataFile << data << std::endl;
    }
}

void RealTimeData::writeToSharedMemory(const std::string &data) {
    std::lock_guard<std::mutex> lock(dataMutex);
    std::memcpy(region.get_address(), data.c_str(), data.size());
}