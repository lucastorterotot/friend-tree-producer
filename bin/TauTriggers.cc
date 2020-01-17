#include "TauAnalysisTools/TauTriggerSFs/interface/TauTriggerSFs2017.h"

#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"
#include "TVector2.h"

#include <math.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <vector>
#include <map>

#include "HiggsAnalysis/friend-tree-producer/interface/HelperFunctions.h"

using boost::starts_with;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  std::string input = "output.root";
  std::string folder = "mt_nominal";
  std::string tree = "ntuple";
  unsigned int first_entry = 0;
  unsigned int last_entry = 9;
  po::variables_map vm;
  po::options_description config("configuration");
  config.add_options()("input",
                       po::value<std::string>(&input)->default_value(input))(
      "folder", po::value<std::string>(&folder)->default_value(folder))(
      "tree", po::value<std::string>(&tree)->default_value(tree))(
      "first_entry",
      po::value<unsigned int>(&first_entry)->default_value(first_entry))(
      "last_entry",
      po::value<unsigned int>(&last_entry)->default_value(last_entry));
  po::store(po::command_line_parser(argc, argv).options(config).run(), vm);
  po::notify(vm);

  // Access input file and tree
  auto in = TFile::Open(input.c_str(), "read");
  auto dir = (TDirectoryFile *)in->Get(folder.c_str());
  auto inputtree = (TTree *)dir->Get(tree.c_str());

  // Read era information from datasets.json
  std::string nickname = filename_from_inputpath(input);
  boost::property_tree::ptree pt;
  boost::property_tree::read_json("HiggsAnalysis/friend-tree-producer/data/input_params/datasets.json", pt);
  int year = pt.get<int>(nickname + ".year");
  std::map<int, std::string> inp_map = {
	  {2016, "TauAnalysisTools/TauTriggerSFs/data/tauTriggerEfficiencies2016KIT_deeptau.root"},
	  {2017, "TauAnalysisTools/TauTriggerSFs/data/tauTriggerEfficiencies2017KIT_deeptau.root"},
	  {2018, "TauAnalysisTools/TauTriggerSFs/data/tauTriggerEfficiencies2018KIT_deeptau.root"},
  }
  // std::string sf_input = "TauAnalysisTools/TauTriggerSFs/data/tauTriggerEfficiencies2017KIT_deeptau.root";
  std::string sf_input = inp_map[year];
  std::cout << "[INFO] Using input file '" << sf_input << "'" << std::endl;
  // Quantities of first lepton
  Float_t pt_1, eta_1, phi_1;
  Int_t dm_1;
  inputtree->SetBranchAddress("pt_1", &pt_1);
  inputtree->SetBranchAddress("eta_1", &eta_1);
  inputtree->SetBranchAddress("phi_1", &phi_1);
  inputtree->SetBranchAddress("decayMode_1", &dm_1);

  // Quantities of second lepton
  Float_t pt_2, eta_2, phi_2;
  Int_t dm_2;
  inputtree->SetBranchAddress("pt_2", &pt_2);
  inputtree->SetBranchAddress("eta_2", &eta_2);
  inputtree->SetBranchAddress("phi_2", &phi_2);
  inputtree->SetBranchAddress("decayMode_2", &dm_2);

  // Initialize output file
  auto outputname =
      outputname_from_settings(input, folder, first_entry, last_entry);
  boost::filesystem::create_directories(filename_from_inputpath(input));
  auto out = TFile::Open(outputname.c_str(), "recreate");
  out->mkdir(folder.c_str());
  out->cd(folder.c_str());

  // Create output tree
  auto triggerfriend = new TTree("ntuple", "Tau trigger friend tree");

  std::vector<std::string> fields;
  boost::split(fields, folder, boost::is_any_of("_"));
  std::string channel =  fields[0];

  std::vector<std::string> work_points = {"vvvloose", "vvloose", "vloose", "loose", "medium", "tight", "vtight", "vvtight"};
  std::vector<double> data_weights;
  std::vector<double> data_weights_up;
  std::vector<double> data_weights_down;
  data_weights.resize(work_points.size(), 1.);
  data_weights_up.resize(work_points.size(), 1.);
  data_weights_down.resize(work_points.size(), 1.);
  for (int i = 0; i < work_points.size(); i++)
  {
	  triggerfriend->Branch("crossTriggerCorrectedDataEfficiencyWeight_" + work_points[i] + "_DeepTau_2", &(data_weights[i]), "crossTriggerCorrectedDataEfficiencyWeight_" + work_points[i] + "_DeepTau_2/F");
	  triggerfriend->Branch("crossTriggerCorrectedDataEfficiencyWeightUp_" + work_points[i] + "_DeepTau_2", &(data_weights_up[i]), "crossTriggerCorrectedDataEfficiencyWeightUp_" + work_points[i] + "_DeepTau_2/F");
	  triggerfriend->Branch("crossTriggerCorrectedDataEfficiencyWeightDown_" + work_points[i] + "_DeepTau_2", &(data_weights_down[i]), "crossTriggerCorrectedDataEfficiencyWeightDown_" + work_points[i] + "_DeepTau_2/F");
  }

  std::vector<double> mc_weights;
  std::vector<double> mc_weights_up;
  std::vector<double> mc_weights_down;
  mc_weights.resize(work_points.size(), 1.);
  mc_weights_up.resize(work_points.size(), 1.);
  mc_weights_down.resize(work_points.size(), 1.);
  for (int i = 0; i < work_points.size(); i++)
  {
	  triggerfriend->Branch("crossTriggerCorrectedMCEfficiencyWeight_" + work_points[i] + "_DeepTau_2", &(mc_weights[i]), "crossTriggerCorrectedMCEfficiencyWeight_" + work_points[i] + "_DeepTau_2/F");
	  triggerfriend->Branch("crossTriggerCorrectedMCEfficiencyWeightUp_" + work_points[i] + "_DeepTau_2", &(mc_weights_up[i]), "crossTriggerCorrectedMCEfficiencyWeightUp_" + work_points[i] + "_DeepTau_2/F");
	  triggerfriend->Branch("crossTriggerCorrectedMCEfficiencyWeightDown_" + work_points[i] + "_DeepTau_2", &(mc_weights_down[i]), "crossTriggerCorrectedMCEfficiencyWeightDown_" + work_points[i] + "_DeepTau_2/F");
  }

  std::vector<double> emb_weights;
  std::vector<double> emb_weights_up;
  std::vector<double> emb_weights_down;
  emb_weights.resize(work_points.size(), 1.);
  emb_weights_up.resize(work_points.size(), 1.);
  emb_weights_down.resize(work_points.size(), 1.);
  for (int i = 0; i < work_points.size(); i++)
  {
	  triggerfriend->Branch("crossTriggerCorrectedEMBEfficiencyWeight_" + work_points[i] + "_DeepTau_2", &(mc_weights[i]), "crossTriggerCorrectedEMBEfficiencyWeight_" + work_points[i] + "_DeepTau_2/F");
	  triggerfriend->Branch("crossTriggerCorrectedEMBEfficiencyWeightUp_" + work_points[i] + "_DeepTau_2", &(mc_weights_up[i]), "crossTriggerCorrectedEMBEfficiencyWeightUp_" + work_points[i] + "_DeepTau_2/F");
	  triggerfriend->Branch("crossTriggerCorrectedEMBEfficiencyWeightDown_" + work_points[i] + "_DeepTau_2", &(mc_weights_down[i]), "crossTriggerCorrectedEMBEfficiencyWeightDown_" + work_points[i] + "_DeepTau_2/F");
  }
  std::vector<double> data_weights1;
  std::vector<double> data_weights1_up;
  std::vector<double> data_weights1_down;
  std::vector<double> emb_weights1;
  std::vector<double> emb_weights1_up;
  std::vector<double> emb_weights1_down;
  std::vector<double> mc_weights1;
  std::vector<double> mc_weights1_up;
  std::vector<double> mc_weights1_down;
  data_weights1.resize(work_points.size(), 1.);
  data_weights1_up.resize(work_points.size(), 1.);
  data_weights1_down.resize(work_points.size(), 1.);
  mc_weights1.resize(work_points.size(), 1.);
  mc_weights1_up.resize(work_points.size(), 1.);
  mc_weights1_down.resize(work_points.size(), 1.);
  emb_weights1.resize(work_points.size(), 1.);
  emb_weights1_up.resize(work_points.size(), 1.);
  emb_weights1_down.resize(work_points.size(), 1.);
  if (channel == "tt")
  {
	  for (int i = 0; i < work_points.size(); i++)
	  {
		  triggerfriend->Branch("crossTriggerCorrectedDataEfficiencyWeight_" + work_points[i] + "_DeepTau_1", &(data_weights1[i]), "crossTriggerCorrectedDataEfficiencyWeight_" + work_points[i] + "_DeepTau_1/F");
		  triggerfriend->Branch("crossTriggerCorrectedDataEfficiencyWeightUp_" + work_points[i] + "_DeepTau_1", &(data_weights1_up[i]), "crossTriggerCorrectedDataEfficiencyWeightUp_" + work_points[i] + "_DeepTau_1/F");
		  triggerfriend->Branch("crossTriggerCorrectedDataEfficiencyWeightDown_" + work_points[i] + "_DeepTau_1", &(data_weights1_down[i]), "crossTriggerCorrectedDataEfficiencyWeightDown_" + work_points[i] + "_DeepTau_1/F");
	  }

	  for (int i = 0; i < work_points.size(); i++)
	  {
		  triggerfriend->Branch("crossTriggerCorrectedMCEfficiencyWeight_" + work_points[i] + "_DeepTau_1", &(mc_weights1[i]), "crossTriggerCorrectedMCEfficiencyWeight_" + work_points[i] + "_DeepTau_1/F");
		  triggerfriend->Branch("crossTriggerCorrectedMCEfficiencyWeightUp_" + work_points[i] + "_DeepTau_1", &(mc_weights1_up[i]), "crossTriggerCorrectedMCEfficiencyWeightUp_" + work_points[i] + "_DeepTau_1/F");
		  triggerfriend->Branch("crossTriggerCorrectedMCEfficiencyWeightDown_" + work_points[i] + "_DeepTau_1", &(mc_weights1_down[i]), "crossTriggerCorrectedMCEfficiencyWeightDown_" + work_points[i] + "_DeepTau_1/F");
	  }

	  for (int i = 0; i < work_points.size(); i++)
	  {
		  triggerfriend->Branch("crossTriggerCorrectedEMBEfficiencyWeight_" + work_points[i] + "_DeepTau_1", &(mc_weights1[i]), "crossTriggerCorrectedEMBEfficiencyWeight_" + work_points[i] + "_DeepTau_1/F");
		  triggerfriend->Branch("crossTriggerCorrectedEMBEfficiencyWeightUp_" + work_points[i] + "_DeepTau_1", &(mc_weights1_up[i]), "crossTriggerCorrectedEMBEfficiencyWeightUp_" + work_points[i] + "_DeepTau_1/F");
		  triggerfriend->Branch("crossTriggerCorrectedEMBEfficiencyWeightDown_" + work_points[i] + "_DeepTau_1", &(mc_weights1_down[i]), "crossTriggerCorrectedEMBEfficiencyWeightDown_" + work_points[i] + "_DeepTau_1/F");
	  }
  }

  // Set up correct readout for channel and wps
  std::map<std::string, std::string> trg_name = {
	  {"et", "etau"},
	  {"mt", "mutau"},
	  {"tt", "ditau"},
  };
  std::map<std::string, TauTriggerSFs2017*> TauSFs;
  for (auto wp: work_points)
  {
	  TauSFs[wp] = new TauTriggerSFs2017(sf_input, sf_input, trg_name[channel], year, wp, "DeepTau");
  }

  // Loop over desired events of the input tree & compute outputs
  for (unsigned int i = first_entry; i <= last_entry; i++) {
    // Get entry
    inputtree->GetEntry(i);

    for (size_t j = 0; j < work_points.size(); j++)
    {
	    data_weights[j] = TauSFs.at(wp)->getTriggerEfficiencyData(pt_2,eta_2,phi_2,dm_2);
	    data_weights_up[j] = TauSFs.at(wp)->getTriggerEfficiencyDataUncertUp(pt_2,eta_2,phi_2,dm_2);
	    data_weights_down[j] = TauSFs.at(wp)->getTriggerEfficiencyDataUncertDown(pt_2,eta_2,phi_2,dm_2);
	    mc_weights[j] = TauSFs.at(wp)->getTriggerEfficiencyMC(pt_2,eta_2,phi_2,dm_2);
	    mc_weights_up[j] = TauSFs.at(wp)->getTriggerEfficiencyMCUncertUp(pt_2,eta_2,phi_2,dm_2);
	    mc_weights_down[j] = TauSFs.at(wp)->getTriggerEfficiencyMCUncertDown(pt_2,eta_2,phi_2,dm_2);
	    emb_weights[j] = TauSFs.at(wp)->getTriggerEfficiencyEMB(pt_2,eta_2,phi_2,dm_2);
	    emb_weights_up[j] = TauSFs.at(wp)->getTriggerEfficiencyEMBUncertUp(pt_2,eta_2,phi_2,dm_2);
	    emb_weights_down[j] = TauSFs.at(wp)->getTriggerEfficiencyEMBUncertDown(pt_2,eta_2,phi_2,dm_2);
    }

    if (channel == "tt")
    {
	    for (size_t j = 0; j < work_points.size(); j++)
	    {
		    data_weights1[j] = TauSFs.at(wp)->getTriggerEfficiencyData(pt_1,eta_1,phi_1,dm_1);
		    data_weights1_up[j] = TauSFs.at(wp)->getTriggerEfficiencyDataUncertUp(pt_1,eta_1,phi_1,dm_1);
		    data_weights1_down[j] = TauSFs.at(wp)->getTriggerEfficiencyDataUncertDown(pt_1,eta_1,phi_1,dm_1);
		    mc_weights1[j] = TauSFs.at(wp)->getTriggerEfficiencyMC(pt_1,eta_1,phi_1,dm_1);
		    mc_weights1_up[j] = TauSFs.at(wp)->getTriggerEfficiencyMCUncertUp(pt_1,eta_1,phi_1,dm_1);
		    mc_weights1_down[j] = TauSFs.at(wp)->getTriggerEfficiencyMCUncertDown(pt_1,eta_1,phi_1,dm_1);
		    emb_weights1[j] = TauSFs.at(wp)->getTriggerEfficiencyEMB(pt_1,eta_1,phi_1,dm_1);
		    emb_weights1_up[j] = TauSFs.at(wp)->getTriggerEfficiencyEMBUncertUp(pt_1,eta_1,phi_1,dm_1);
		    emb_weights1_down[j] = TauSFs.at(wp)->getTriggerEfficiencyEMBUncertDown(pt_1,eta_1,phi_1,dm_1);
	    }
    }

    // Fill output tree
    triggerfriend->Fill();
  }

  // Clean up
  for (std::map<std::string, TauTriggerSFs2017*>::iterator trig_map_it = TauSFs.begin(); trig_map_it != TauSFs.end(); trig_map_it++)
  {
  	delete trig_map_it->second;
  }
  // Fill output file
  out->cd(folder.c_str());
  triggerfriend->Write("", TObject::kOverwrite);
  out->Close();
  in->Close();

  return 0;
}
