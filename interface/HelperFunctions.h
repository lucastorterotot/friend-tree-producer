#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::string folder_to_channel(std::string foldername)
{
    std::vector<std::string> foldername_split;
    boost::split(foldername_split,foldername, [](char c){return c == '_';});
    return foldername_split.at(0);
}

std::string filename_from_inputpath(std::string input)
{
    std::vector<std::string> path_split;
    boost::split(path_split,input, [](char c){return c == '/';});
    std::string filename = path_split.at(path_split.size() - 1);
    boost::replace_all(filename, ".root", "");
    return filename;
}

std::string outputname_from_settings(std::string input, std::string folder, unsigned int first_entry, unsigned int last_entry,  fs::path output_dir="", bool organize_outputs = true)
{
    std::string filename = filename_from_inputpath(input);
    fs::path full_path = "";
    if(organize_outputs) {
        std::string outputname = filename + "/" + filename + "_" + folder + "_" + std::to_string(first_entry) + "_" + std::to_string(last_entry) + ".root";
        full_path = output_dir / outputname;
    }
    else {
        std::string outputname = filename + "_" + folder + "_" + std::to_string(first_entry) + "_" + std::to_string(last_entry) + ".root";
        full_path = outputname;
    }
    return full_path.string();
}

const auto default_float = -10.f;
