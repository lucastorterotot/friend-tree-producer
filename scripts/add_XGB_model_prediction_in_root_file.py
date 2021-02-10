#!/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt/bin/python3
# -*- coding: utf-8 -*-

"""
    Example:
        python HiggsAnalysis/friend-tree-producer/scripts/add_ML_models_prediction_in_root_file.py \
            --input <path to root file> \
            --XGBs <path to XGB json files> \
            --output-dir  <out dir> \
            --dry
"""

import sys, os
import argparse
import logging

sys.path = [
    '/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt/lib/python3.7/site-packages',
] + sys.path
import numpy as np

sys.path = [
    '{}/.local/lib/python3.7/site-packages/'.format(os.environ["HOME"]),
    '/home/ltortero/.local/lib/python3.7/site-packages/',
] + sys.path
from xgboost import XGBRegressor
import uproot
import pandas
import array

try:
    from ROOT import TFile, TDirectoryFile, TTree
    from ROOT.TObject import kOverwrite
except:
    print("Try to load ROOT from elsewhere...")
    sys.path = [
        '/cvmfs/sft.cern.ch/lcg/views/LCG_93python3/x86_64-centos7-gcc62-opt/lib',
    ] + sys.path
    from ROOT import TFile, TDirectoryFile, TTree
    from ROOT.TObject import kOverwrite

logger = logging.getLogger()

os.environ['PYTHONPATH'] = '/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt/lib/python3.7/site-package:' + os.environ['PYTHONPATH']

sys.path = [
    '/cvmfs/sft.cern.ch/lcg/views/LCG_96bpython3/x86_64-ubuntu1804-gcc8-opt/lib',
    '/cvmfs/sft.cern.ch/lcg/views/LCG_96bpython3/x86_64-ubuntu1804-gcc8-opt/lib/python3.6/site-packages',
] + sys.path

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

    parser.add_argument("--XGBs", required=False, type=str, help="Input XGB json files.", default="")

    parser.add_argument(
        "--tree", default="ntuple", type=str, help="Name of the root tree."
    )

    parser.add_argument(
        "--channels", default="all", type=str, help="Channels to process, comma separated."
    )

    parser.add_argument(
        "--categories", default="all", type=str, help="Categories to process, comma separated OR 'all'."
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

class XGB_model_from_json(object):
    
    def __init__(self, json_file):
        name = json_file.replace("/work/ltortero/ML_models/", "")
        name = name.replace("trained_NNs_FastSim/", "")
        name = name.replace("trained_xgboosts_FastSim/", "")
        name = name.replace("/", "_")
        name = name.replace('.json', '')
        name = "DNN_" + name
        name = name.replace("-","_")
        self.name = name
        
        # load json and create model
        loaded_model = XGBRegressor()
        loaded_model.load_model(json_file)
    
        print("Loaded XGBRegressor model from disk:")
        print("\t{}".format(json_file))

        self.model = loaded_model
            
        # load list of inputs for the model
        sys.path.insert(0, json_file.rstrip(json_file.split('/')[-1]))
        import inputs_for_models_in_this_dir
        reload(inputs_for_models_in_this_dir) # avoid being stuck with previous versions
        this_model_inputs = inputs_for_models_in_this_dir.inputs
        this_model_inputs = [i if i not in var_names_at_KIT.keys() else var_names_at_KIT[i] for i in this_model_inputs]
        self.inputs = this_model_inputs

    def predict(self, filtered_df):
        return self.model.predict(np.r_[filtered_df[self.inputs]])

def main(args):
    print(args)

    channels = args.channels.split(',')
    categories = args.categories.split(',')

    nickname = os.path.basename(args.input).replace(".root", "")

    XGB_jsons = args.XGBs.split(',')
    XGB_jsons = [f for f in XGB_jsons if f != ""]

    models = {}
    inputs = []
    for XGB_json in XGB_jsons:
        XGB_object = XGB_model_from_json(XGB_json)
        models[XGB_object.name] = XGB_object
        inputs += XGB_object.inputs
    
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

    root_file_in = uproot.open(root_file_input)

    if 'all' in channels:
        channels = set([k.split('_')[0] for k in root_file_in.keys()])
    if 'all' in categories:
        categories = set([k.split('_')[-1].split(';')[0] for k in root_file_in.keys() if any([c in k for c in channels])])

    if not args.dry:
        root_file_old = None
        if not args.recreate:
            os.system("mv {} {}_to_update".format(root_file_output, root_file_output))
            root_file_old = TFile("{}_to_update".format(root_file_output), 'read')
        root_file_out = TFile(root_file_output, 'recreate')
        print("Opened new file")
    first_pass = True

    for channel in channels:
        for cat in categories:
            rootdirname = '{}_{}'.format(channel, cat)
            if rootdirname not in root_file_in.keys() and "{};1".format(rootdirname) not in root_file_in.keys():
                continue
            if rootdirname != args.pipeline and args.pipeline != None:
                continue

            print('process pipeline: %s_%s' % (channel, cat))

            if not first_pass and not args.dry:
                root_file_out = TFile(root_file_output, 'update')
            first_pass = False

            if not args.dry:
                rootdir_old = False
                if root_file_old:
                    rootdir_old = root_file_old.GetDirectory(rootdirname)
                if not rootdir_old:
                    already_rootdir = False
                else:
                    already_rootdir = True
                rootdir = TDirectoryFile(rootdirname, rootdirname)
                rootdir.cd()
                tree_old = False
                if already_rootdir:
                    if not args.recreate:
                        tree_old = rootdir_old.Get(args.tree)
                tree = TTree(args.tree, args.tree)

                old_models = []
                if tree_old:
                    old_models = [model.GetName() for model in [tree_old.GetListOfBranches()][0]]
                if len(old_models) > 0:
                    root_file_out_old = uproot.open("{}_to_update".format(root_file_output))

                models = {i:models[i] for i in models if i not in old_models}
                all_models = old_models + [k for k in models]

                leafValues = {}
                for model in all_models:
                    leafValues[model] = array.array("f", [0])

            if args.pandas:
                df = root_file_in[rootdirname][args.tree].pandas.df()
                if tree_old:
                    df_old = root_file_out_old[rootdirname][args.tree].pandas.df()
            else:
                _df = root_file_in[rootdirname][args.tree].arrays()
                df = pandas.DataFrame()
                keys_to_export = set(inputs+["pt_1", "pt_2", "phi_1", "phi_2"])
                for key in ["N_neutrinos_reco", "mt_tt"]:
                    if key in keys_to_export:
                        keys_to_export.remove(key)
                for k in keys_to_export:
                    df[k] = _df[str.encode(k)]
                if tree_old:
                    _df_old = root_file_out_old[rootdirname][args.tree].arrays()
                    df_old = pandas.DataFrame()
                    keys_to_export = old_models
                    for k in keys_to_export:
                        df_old[k] = _df_old[str.encode(k)]

            df["mt_tt"] = (2*df["pt_1"]*df["pt_2"]*(1-np.cos(df["phi_1"]-df["phi_2"])))**.5

            df["N_neutrinos_reco"] = N_neutrinos_in_channel[channel] * np.ones(len(df[inputs[0]]), dtype='int')
    
            # remove values set at -10 by default to match training settings
            for variable in ["jpt_r", "jeta_r", "jphi_r", "Njet_r"]:
                if variable in inputs:
                    df[variable].values[df[variable].values < 0] = 0
    
            for model in models:
                print("Predicting with {}...".format(model))
                df[model] = models[model].predict(df)

            if not args.dry:
                print("Filling new branch in tree...")
                for model in all_models:
                    newBranch = tree.Branch(
                        model,
                        leafValues[model],
                        "prediction/F"
                    )
                first_entry = args.first_entry
                last_entry = len(df[model].values)
                if args.last_entry > first_entry and args.last_entry < len(df[model].values):
                    last_entry = args.last_entry
                for k in range(first_entry, last_entry +1):
                    for model in all_models:
                        if model in old_models:
                            leafValues[model][0] = df_old[model].values[k]
                        else:
                            leafValues[model][0] = df[model].values[k]
                    tree.Fill()
                print("Filled.")
                
                rootdir.Remove(rootdir.Get(args.tree))

                tree.Write(args.tree, kOverwrite)
                root_file_out.Close()
                os.system("rm -rf {}_to_update".format(root_file_output))

if __name__ == "__main__":
    args = parse_arguments()

    if args.enable_logging:
        setup_logging(
            "add_XGB_model_prediction_in_root_file_%s_%s_%s_%s.log"
            % (
                os.path.basename(args.input).replace(".root", ""),
                args.folder,
                args.first_entry,
                args.last_entry,
            ),
            logging.WARNING,
        )

    main(args)
