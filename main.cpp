//
// Created by songtf on 20-10-15.
//


#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <algorithm>


void leftTrim(std::string &s)
{
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

void rightTrim(std::string &s)
{
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
    leftTrim(s);
    rightTrim(s);
    return s;
}


std::vector<std::map<std::string, std::string>> parseConfigFile(const std::string cfgFilePath)
{
    std::ifstream file(cfgFilePath);
    std::string line;
    std::vector<std::map<std::string, std::string>> blocks;
    std::map<std::string, std::string> block;

    while (getline(file, line))
    {
        if (line.size() == 0) continue;
        if (line.front() == '#') continue;
        line = trim(line);
        if (line.front() == '[')
        {
            if (block.size() > 0)
            {
                blocks.push_back(block);
                block.clear();
            }
            std::string key = "type";
            std::string value = trim(line.substr(1, line.size() - 2));
            block.insert(std::pair<std::string, std::string>(key, value));
        } else
        {
            size_t cpos = line.find('=');
            std::string key = trim(line.substr(0, cpos));
            std::string value = trim(line.substr(cpos + 1));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
    }
    blocks.push_back(block);
    return blocks;
}


int main()
{
//    // Yolov3配置网路
//    Config config_v3;
//    config_v3.net_type = YOLOV3;
//    config_v3.file_model_cfg = "";
//    config_v3.file_model_weights = "";
//    config_v3.calibration_image_list_file_txt = "";
//    config_v3.inference_precision = FP32;
//    config_v3.gpu_id = 0;
//
//
//
//    std::make_shared<Detector> detector;
//    detector->init(config_v3);
//
//    //


    std::string cfgFilePath = "/home/songtf/project/yolo-tensorrt/configs/yolov3.cfg";
    std::vector<std::map<std::string, std::string>> m_configBlocks = parseConfigFile(cfgFilePath);

    for (uint32_t i = 0; i < m_configBlocks.size(); ++i)
    {
        std::cout << m_configBlocks.at(i).at("type") << std::endl;
        if (m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end())
        {
            std::cout << "BatchNormalize True" << std::endl;
        }
        if (m_configBlocks.at(i).find("activation") != m_configBlocks.at(i).end())
        {
            std::cout << m_configBlocks[i]["activation"] << std::endl;
        }
    }


    return 0;
}