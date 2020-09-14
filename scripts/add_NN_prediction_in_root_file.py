#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#__requires__= "Keras==2.2.4"

"""
    Example:
        python HiggsAnalysis/friend-tree-producer/scripts/ add_NN_prediction_in_root_file.py \
            --input <path to root file> \
            --NN <path to NN json file> \
            --output-dir  <out dir> \
            --dry
"""
import os
import sys
import json
import yaml
import ROOT
import numpy
import copy
from array import array
import six
import argparse
import logging
import pkg_resources
#pkg_resources.require("Keras==2.2.4")
from keras.models import model_from_json
import uproot
import numpy as np

logger = logging.getLogger()

pipelines = ["nominal"]


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
        "--enable-logging",
        action="store_true",
        help="Enable loggging for debug purposes.",
    )
    parser.add_argument(
        "--cmsswbase",
        default=os.environ["CMSSW_BASE"],
        help="Set path for to local cmssw for submission with Grid-Control",
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

    return parser.parse_args()

def main(args):
    print(args)
    nickname = os.path.basename(args.input).replace(".root", "")
    cmsswbase = args.cmsswbase
    input_json = args.NN
    
    NNname = input_json.split('/')[-1].replace('.json', '').replace("-","_")

    # load json and create model
    NN_weights_path_and_file = input_json.split('/')
    NN_weights_path_and_file[-1] = "NN_weights-{}".format(NN_weights_path_and_file[-1].replace('.json', '.h5'))
    NN_weights_file = "/".join(NN_weights_path_and_file)

    json_file = open(input_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(NN_weights_file)
    print("Loaded model from disk:")
    print("\t{}".format(input_json))

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
    df = root_file_in['events'].pandas.df()
    
    df["mt_tt"] = (2*df["l1_pt"]*df["l2_pt"]*(1-np.cos(df["l1_phi"]-df["l2_phi"])))**.5
    
    inputs = [
        "tau1_pt",
        "tau1_eta",
        "tau1_phi",
        "tau2_pt",
        "tau2_eta",
        "tau2_phi",
        # "jet1_pt",
        # "jet1_eta",
        # "jet1_phi",
        # "jet2_pt",
        # "jet2_eta",
        # "jet2_phi",
        # "recoil_pt",
        # "recoil_eta",
        # "recoil_phi",
        "MET_pt",
        "MET_phi",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        # "MET_significance",
        "mT1",
        "mT2",
        "mTtt",
        "mTtot",
    ]
    
    df["predictions_{}".format(NNname)] = loaded_model.predict(df[inputs])
    
    tree_dtype = {}
    tree_data = {}
    for b in df.keys():
        tree_dtype[b] = df[b].dtype.name
        if tree_dtype[b] == 'uint64':
            tree_dtype[b] = 'int64'
        tree_data[b] = np.array(df[b])

    root_file_out = uproot.recreate(root_file_output)
    print("Opened new file")
    root_file_out.newtree('events', tree_dtype)
    print("Created new tree")
    root_file_out['events'].extend(tree_data)
    print("New tree filled")

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
