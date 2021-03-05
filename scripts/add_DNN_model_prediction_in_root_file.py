#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys, os
sys.path = ['{}/src/HiggsAnalysis/friend-tree-producer/python/'.format(os.environ["CMSSW_BASE"])] + sys.path

"""
    Example:
        python HiggsAnalysis/friend-tree-producer/scripts/ add_DNN_model_prediction_in_root_file.py \
            --input <path to root file> \
            --DNNs <path to DNNs json file> \
            --output-dir  <out dir> \
            --dry
"""

import numpy as np
import argparse
import logging
from keras.models import model_from_json
import uproot
import pandas
from ROOT import TFile, TDirectoryFile, TTree
from ROOT.TObject import kOverwrite
import array

logger = logging.getLogger()

def setup_logging(output_file, level=logging.DEBUG):
    logger.setLevel(level)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(output_file, "w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="create friend trees for electron scale factors form a given RooWorkspace"
    )
    parser.add_argument("--input", required=True, type=str, help="Input root file.")

    parser.add_argument("--DNNs", required=True, type=str, help="Input DNNs json file.")

    parser.add_argument(
        "--tree", default="ntuple", type=str, help="Name of the root tree."
    )

    parser.add_argument(
        "--channels", default="all", type=str, help="Channels to process, comma separated."
    )

    parser.add_argument(
        "--categories", default="all", type=str, help="Categories to process, comma separated."
    )

    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable loggging for debug purposes.",
    )

    parser.add_argument(
        "--first-entry",
        "--first_entry",
        "--start",
        default=0,
        type=int,
        help="Index of first event to process.",
    )
    parser.add_argument(
        "--last-entry",
        "--last_entry",
        "--end",
        default=-1,
        type=int,
        help="Index of last event to process.",
    )

    parser.add_argument(
        "--pipeline",
        "--pipelines",
        "--folder",
        nargs="?",
        default=None,
        type=str,
        help="Directory within rootfile.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Tag of output files."
    )

    parser.add_argument("--config", nargs="?", type=str, default=None, help="Config")

    parser.add_argument("--dry", action="store_true", default=False, help="dry run")

    parser.add_argument("--recreate", action="store_true", default=False, help="Whether to just update or fully-recreate the friend tree.")

    parser.add_argument("--pandas", action="store_true", default=False, help="Whether to use arrays or pandas dataframe with uproot")

    parser.add_argument("--organize_outputs", default=False, help="Just compatibility: Remote access currently only works for SVFit computation")

    return parser.parse_args()

var_names_at_KIT = {
    "tau1_pt_reco" : "pt_1",
    "tau1_eta_reco" : "eta_1",
    "tau1_phi_reco" : "phi_1",
    "tau2_pt_reco" : "pt_2",
    "tau2_eta_reco" : "eta_2",
    "tau2_phi_reco" : "phi_2",
    "jet1_pt_reco" : "jpt_1",
    "jet1_eta_reco" : "jeta_1",
    "jet1_phi_reco" : "jphi_1",
    "jet2_pt_reco" : "jpt_2",
    "jet2_eta_reco" : "jeta_2",
    "jet2_phi_reco" : "jphi_2",
    "remaining_jets_pt_reco" : "jpt_r",
    "remaining_jets_eta_reco" : "jeta_r",
    "remaining_jets_phi_reco" : "jphi_r",
    "remaining_jets_N_reco" : "Njet_r",
    "MET_pt_reco" : "met",
    "MET_phi_reco" : "metphi",
    "MET_covXX_reco" : "metcov00",
    "MET_covXY_reco" : "metcov01",
    "MET_covYY_reco" : "metcov11",
    "mT1_reco" : "mt_1",
    "mT2_reco" : "mt_2",
    "mTtt_reco" : "mt_tt",
    "mTtot_reco" : "mt_tot",
    "PuppiMET_pt_reco" : "puppimet",
    "PuppiMET_phi_reco" : "puppimetphi",
    "PuppimT1_reco" : "mt_1_puppi",
    "PuppimT2_reco" : "mt_2_puppi",
    "PuppimTtt_reco" : "mt_tt",
    "PuppimTtot_reco" : "mt_tot_puppi",
    "PU_npvsGood_reco" : "npv",
}

N_neutrinos_in_channel = {
    "tt" : 2,
    "mt" : 3,
    "et" : 3,
    "mm" : 4,
    "em" : 4,
    "ee" : 4,
}

env={}
for k in os.environ.keys():
    env[k] = os.environ[k]

class DNN_model_from_json(object):
    
    def __init__(self, json_file):
        # load json and create model
        NN_weights_path_and_file = json_file.split('/')
        NN_weights_path_and_file[-1] = "NN_weights-{}".format(NN_weights_path_and_file[-1].replace('.json', '.h5'))
        NN_weights_file = "/".join(NN_weights_path_and_file)

        json_file_ = open(json_file, 'r')
        loaded_model_json = json_file_.read()
        json_file_.close()
        loaded_model = model_from_json(loaded_model_json)
    
        # load weights into new model
        loaded_model.load_weights(NN_weights_file)
        print("Loaded DNN model from disk:")
        print("\t{}".format(json_file))

        self.model = loaded_model

        # load list of inputs for the model
        sys.path.insert(0, json_file.rstrip(json_file.split('/')[-1]))
        import inputs_for_models_in_this_dir
        reload(inputs_for_models_in_this_dir) # avoid being stuck with previous versions
        this_model_inputs = inputs_for_models_in_this_dir.inputs
        this_model_inputs = [i if i not in var_names_at_KIT.keys() else var_names_at_KIT[i] for i in this_model_inputs]
        self.inputs = this_model_inputs
            
    def predict(self, evt, channel):
        # Get the inputs from tree
        df = {}
        for input in self.inputs:
            if input not in ["mt_tt", "N_neutrinos_reco"]:
                df[input] = evt.GetLeaf(input).GetValue(0)
        # get mt_tt as defined for training
        df["mt_tt"] = (2*df["pt_1"]*df["pt_2"]*(1-np.cos(df["phi_1"]-df["phi_2"])))**.5
        # derive N neutrinos
        df["N_neutrinos_reco"] = N_neutrinos_in_channel[channel]
        # Set -10 to 0 for variable in ["jpt_r", "jeta_r", "jphi_r", "Njet_r"] as defined in training
        for variable in ["jpt_r", "jeta_r", "jphi_r", "Njet_r"]:
            if variable in self.inputs:
                if df[input] == -10:
                    df[input] = 0
        return self.model.predict(np.array([[df[input] for input in self.inputs]]))

def main(args):
    print(args)

    channels = args.channels.split(',')
    categories = args.categories.split(',')

    nickname = os.path.basename(args.input).replace(".root", "")

    DNN_jsons = args.DNNs.split(',')
    DNN_jsons = [f for f in DNN_jsons if f != ""]

    models = {}
    inputs = []
    model_number=1
    for DNN_json in DNN_jsons:
        DNN_object = DNN_model_from_json(DNN_json)
        models["DNN{}".format(model_number)] = DNN_object
        model_number += 1
        inputs += DNN_object.inputs

    # load root file and create friend tree
    root_file_input = args.input
    output_path = os.path.join(args.output_dir, nickname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    root_file_output = os.path.join(
        output_path,
        "_".join(
            filter(
                None,
                [
                    nickname,
                    args.pipeline,
                    str(args.first_entry),
                    str(args.last_entry),
                ],
            )
        )
        + ".root",
    )

    root_file_in = TFile.Open(root_file_input, 'read')

    if 'all' in channels:
        channels = set([k.GetName().split('_')[0] for k in root_file_in.GetListOfKeys()])
    if 'all' in categories:
        categories = set([k.GetName().split('_')[-1] for k in root_file_in.GetListOfKeys() if any([c == k.GetName().split('_')[0] for c in channels])])

    if not args.dry:
        root_file_out = TFile.Open(root_file_output, 'recreate')
        print("Opened new file")
    first_pass = True

    for channel in channels:
        for cat in categories:
            rootdirname = '{}_{}'.format(channel, cat)
            print(rootdirname)
            if rootdirname not in [k.GetName() for k in root_file_in.GetListOfKeys()]:
                continue
            if rootdirname != args.pipeline and args.pipeline != None:
                continue

            print('process pipeline: %s_%s' % (channel, cat))

            if not first_pass and not args.dry:
                root_file_out = TFile.Open(root_file_output, 'update')
            first_pass = False

            if not args.dry:
                rootdir = TDirectoryFile(rootdirname, rootdirname)
                rootdir.cd()
                tree = TTree(args.tree, args.tree)

                leafValues = {}
                for model in models:
                    leafValues[model] = array.array("f", [0])

            tree_from_root_file_in = root_file_in.Get(rootdirname).Get(args.tree)
            if not args.dry:
                print("Filling new branch in tree...")
                for model in models:
                    print(model)
                    newBranch = tree.Branch(
                        model,
                        leafValues[model],
                        "{}/F".format(model)
                    )
                first_entry = args.first_entry
                last_entry = tree_from_root_file_in.GetEntries()
                if args.last_entry > first_entry and args.last_entry < last_entry:
                    last_entry = args.last_entry
                k = 0
                for evt in tree_from_root_file_in:
                    if k >= first_entry and k <= last_entry:
                        for model in models:
                            leafValues[model][0] = models[model].predict(evt, channel)
                        tree.Fill()
                    k += 1
                print("Filled.")

            if not args.dry:
                tree.Write(args.tree, kOverwrite)
                root_file_out.Close()

    print("Done")
    # if not args.dry:
    #     os.system(
    #         "mv {f} . ; rmdir $(dirname {f})".format(
    #             f = root_file_output,
    #         )
    #     )

if __name__ == "__main__":
    args = parse_arguments()

    if args.enable_logging:
        setup_logging(
            "add_DNN_model_prediction_in_root_file_%s_%s_%s_%s.log"
            % (
                os.path.basename(args.input).replace(".root", ""),
                args.folder,
                args.first_entry,
                args.last_entry,
            ),
            logging.WARNING,
        )

    main(args)
