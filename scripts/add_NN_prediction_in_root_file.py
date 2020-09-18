#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
    Example:
        python HiggsAnalysis/friend-tree-producer/scripts/ add_NN_prediction_in_root_file.py \
            --input <path to root file> \
            --NN <path to NN json file> \
            --output-dir  <out dir> \
            --dry
"""
import os
import numpy as np
import argparse
import logging
from keras.models import model_from_json
import uproot
import pandas
from ROOT import TFile, TDirectoryFile, TTree
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

    parser.add_argument("--NN", required=True, type=str, help="Input NN json file.")

    parser.add_argument(
        "--tree", default="ntuple", type=str, help="Name of the root tree."
    )

    parser.add_argument(
        "--channels", default="mt", type=str, help="Channels to process, comma separated."
    )

    parser.add_argument(
        "--categories", default="nominal", type=str, help="Categories to process, comma separated OR 'all'."
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

class NN_model_from_json(object):
    
    def __init__(self, json_file):
        self.name = json_file.split('/')[-1].replace('.json', '').replace("-","_")
        
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
        print("Loaded model from disk:")
        print("\t{}".format(json_file))

        self.NN = loaded_model
            
    def predict(self, filtered_df):
        return self.NN.predict(filtered_df)

def main(args):
    print(args)

    channels = args.channels.split(',')
    categories = args.categories.split(',')

    nickname = os.path.basename(args.input).replace(".root", "")

    NN_jsons = args.NN.split(',')
    NNs = {}
    for NN_json in NN_jsons:
        NN_object = NN_model_from_json(NN_json)
        NNs[NN_object.name] = NN_object
    
    inputs = [
        "pt_1",
        "eta_1",
        "phi_1",
        "pt_2",
        "eta_2",
        "phi_2",
        # "jpt_1",
        # "jeta_1",
        # "jphi_1",
        # "jpt_2",
        # "jeta_2",
        # "jphi_2",
        # "recoil_pt",
        # "recoil_eta",
        # "recoil_phi",
        "met",
        "metphi",
        "metcov00",
        "metcov01",
        "metcov11",
        # "MET_significance",
        "mT1",
        "mT2",
        "mTtt",
        "mTtot",
    ]

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
        channels = set([k.split('_')[0] for k in root_file_in.keys() if 'nominal' in k])
    if 'all' in categories:
        categories = set([k.split('_')[-1] for k in root_file_in.keys() if any([c in k for c in channels])])

    if not args.dry:
        if args.recreate:
            root_file_out = TFile(root_file_output, 'recreate')
        else:
            root_file_out = TFile(root_file_output, 'update')
        print("Opened new file")
    first_pass = True

    for channel in channels:
        for cat in categories:
            print('process pipeline: %s_%s' % (channel, cat))

            if not first_pass and not args.dry:
                root_file_out = TFile(root_file_output, 'update')
            first_pass = False

            rootdirname = '{}_{}'.format(channel, cat)
            if not args.dry:
                rootdir = root_file_out.GetDirectory(rootdirname)
                if not rootdir:
                    already_rootdir = False
                    rootdir = TDirectoryFile(rootdirname, rootdirname)
                else:
                    already_rootdir = True
                rootdir.cd()
                if not args.recreate and already_rootdir:
                    rootdir.Remove(rootdir.Get(args.tree))
                tree = TTree(args.tree, args.tree)
                leafValues = {}
                for NN in NNs:
                    leafValues[NN] = array.array("f", [0])

            if args.pandas:
                df = root_file_in[rootdirname][args.tree].pandas.df()
            else:
                _df = root_file_in[rootdirname][args.tree].arrays()
                df = pandas.DataFrame()
                keys_to_export = set(inputs+["pt_1", "pt_2", "phi_1", "phi_2", "met", "metphi"])
                for key in ["mTtt", "mT1", "mT2", "mTtot"]:
                    keys_to_export.remove(key)
                for k in keys_to_export:
                    df[k] = _df[k]

            df["mTtt"] = (2*df["pt_1"]*df["pt_2"]*(1-np.cos(df["phi_1"]-df["phi_2"])))**.5
            for leg in [1,2]:
                df["mT{}".format(leg)] = (2*df["pt_{}".format(leg)]*df["met"]*(1-np.cos(df["phi_{}".format(leg)]-df["metphi"])))**.5
            df["mTtot"] = (df["mT1"]**2+df["mT2"]**2+df["mTtt"]**2)**.5
    
            for NN in NNs:
                df["predictions_{}".format(NN)] = NNs[NN].predict(df[inputs])

            if not args.dry:
                print("Filling new branch in tree...")
                for NN in NNs:
                    newBranch = tree.Branch(
                        "predictions_{}".format(NN),
                        leafValues[NN],
                        "predictions_{}/F".format(NN)
                    )
                for k in range(len(df["predictions_{}".format(NN)].values)):
                    for NN in NNs:
                        leafValues[NN][0] = df["predictions_{}".format(NN)].values[k]
                    tree.Fill()
                print("Filled.")

                tree.Write()
                root_file_out.Close()

if __name__ == "__main__":
    args = parse_arguments()

    if args.enable_logging:
        setup_logging(
            "add_NN_prediction_in_root_file_%s_%s_%s_%s.log"
            % (
                os.path.basename(args.input).replace(".root", ""),
                args.folder,
                args.first_entry,
                args.last_entry,
            ),
            logging.WARNING,
        )

    main(args)
