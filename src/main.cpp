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

#include <memory>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <ctime>
#include "Logger.h"
#include "RealTimeData.h"

std::string generateLogFileName() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << "logs/realtime_data_"
       << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%H-%M-%S")
       << ".log";
    return ss.str();
}

int main() {
    std::string logFileName = generateLogFileName();
    std::shared_ptr<Logger> logger = std::make_shared<Logger>(logFileName);
    STX_LOGI(logger, "Start main");

    std::shared_ptr<RealTimeData> dataCollector = std::make_shared<RealTimeData>(logger);

    dataCollector->start();

    return 0;
}