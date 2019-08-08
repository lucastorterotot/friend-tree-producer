#include "lwtnn/LightweightNeuralNetwork.hh"
#include "lwtnn/parse_json.hh"
#include <fstream>

#include "TFile.h"
#include "TH2D.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TVector2.h"
#include "Math/LorentzVector.h"
#include "Math/PtEtaPhiM4D.h"
#include "Math/PxPyPzM4D.h"
#include "Math/Vector4Dfwd.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <iostream>

#include "HiggsAnalysis/friend-tree-producer/interface/HelperFunctions.h"

using boost::starts_with;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  std::string input = "output.root";
  std::vector<std::string> input_friends  = {};
  std::string folder = "mt_nominal";
  std::string tree = "ntuple";
  std::string datasets = std::string(std::getenv("CMSSW_BASE"))+"/src/HiggsAnalysis/friend-tree-producer/data/input_params/datasets.json";
  std::string weight_directory = std::string(std::getenv("CMSSW_BASE"))+"/src/HiggsAnalysis/friend-tree-producer/data/zptm_reweighting/";
  unsigned int first_entry = 0;
  unsigned int last_entry = 9;
  po::variables_map vm;
  po::options_description config("configuration");
  config.add_options()
     ("input",         po::value<std::string>(&input)->default_value(input))
     ("input_friends", po::value<std::vector<std::string>>(&input_friends)->multitoken())
     ("folder",        po::value<std::string>(&folder)->default_value(folder))
     ("tree",          po::value<std::string>(&tree)->default_value(tree))
     ("first_entry",   po::value<unsigned int>(&first_entry)->default_value(first_entry))
     ("last_entry",    po::value<unsigned int>(&last_entry)->default_value(last_entry))
     ("datasets",  po::value<std::string>(&datasets)->default_value(datasets));
  po::store(po::command_line_parser(argc, argv).options(config).run(), vm);
  po::notify(vm);
  // Add additional info inferred from options above
  std::vector<std::string> folder_split;
  boost::split(folder_split, folder, boost::is_any_of("_"));
  std::string channel = folder_split.at(0);
  std::vector<std::string> input_split;
  boost::split(input_split, input, boost::is_any_of("/"));
  std::string nick = input_split.end()[-2];

  // Load datasets.json to determine year of the sample per nick
  boost::property_tree::ptree datasets_json;
  boost::property_tree::json_parser::read_json(datasets,datasets_json);
  int year = datasets_json.get_child(nick).get<int>("year");

  // Access input file and tree
  auto in = TFile::Open(input.c_str(), "read");
  auto dir = (TDirectoryFile *)in->Get(folder.c_str());
  auto inputtree = (TTree *)dir->Get(tree.c_str());
  for(std::vector<std::string>::const_iterator s = input_friends.begin(); s != input_friends.end(); ++s)
  {
    inputtree->AddFriend((folder+"/"+tree).c_str(), (&(*s))->c_str());
  }

  // Initialize weight histogram
  std::string weight_filename = weight_directory+"zptm_weights_" + std::to_string(year) + "_kit.root";
  TFile* weight_file = TFile::Open(weight_filename.c_str(), "read");
  TH2D* weight_hist = (TH2D*) weight_file->Get("zptmass_histo");

  // Initialize inputs
  Float_t genbosonmass, genbosonpt; 
  inputtree->SetBranchAddress("genbosonmass", &genbosonmass);
  inputtree->SetBranchAddress("genbosonpt",   &genbosonpt);

  // Initialize output file
  auto outputname =
      outputname_from_settings(input, folder, first_entry, last_entry);
  boost::filesystem::create_directories(filename_from_inputpath(input));
  auto out = TFile::Open(outputname.c_str(), "recreate");
  out->mkdir(folder.c_str());
  out->cd(folder.c_str());

  // Create output tree
  auto zptm_friend = new TTree("ntuple", "Z(Pt,Mass) weight friend tree");

  // Initialize outputs for the tree
  Float_t zptmass_weight = 1.0; // default value in case no reweighting is needed
  zptm_friend->Branch("zPtMassWeightKIT", &(zptmass_weight), "zPtMassWeightKIT/F");

  // Loop over desired events of the input tree & compute outputs
  for (unsigned int i = first_entry; i <= last_entry; i++) {
    // Get entry
    inputtree->GetEntry(i);

    if(genbosonmass >= 50.0) // no reweighting for events with mass < 50.0 GeV
    {
        // Determine correction weight
        double epsilon = 0.0001;
        auto maxmass = weight_hist->GetXaxis()->GetXmax();
        auto maxpt   = weight_hist->GetYaxis()->GetXmax();
        if(genbosonmass >= maxmass) genbosonmass = maxmass - epsilon;
        if(genbosonpt >= maxpt) genbosonpt = maxpt - epsilon;
        zptmass_weight = weight_hist->GetBinContent(weight_hist->FindBin(genbosonmass, genbosonpt));
        //std::cout << "Determined weight: " << zptmass_weight << " mass: " << genbosonmass << " pt: " << genbosonpt << std::endl;
    }

    // Fill output tree
    zptm_friend->Fill();
  }

  // Fill output file
  out->cd(folder.c_str());
  zptm_friend->Write("", TObject::kOverwrite);
  out->Close();
  in->Close();

  return 0;
}
