#!/usr/bin/env python3
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
import numpy as np
import argparse
import logging
#pkg_resources.require("Keras==2.2.4")
from keras.models import model_from_json
import uproot

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
    df = root_file_in['mt_nominal']['ntuple'].pandas.df()
    
    df["mTtt"] = (2*df["pt_1"]*df["pt_2"]*(1-np.cos(df["phi_1"]-df["phi_2"])))**.5
    for leg in [1,2]:
        df["mT{}".format(leg)] = (2*df["pt_{}".format(leg)]*df["met"]*(1-np.cos(df["phi_{}".format(leg)]-df["metphi"])))**.5
    df["mTtot"] = (df["mT1"]**2+df["mT2"]**2+df["mTtt"]**2)**.5
    
    inputs = [
        "pt_1",
        "eta_1",
        "phi_1",
        "pt_2",
        "eta_2",
        "phi_2",
        "jpt_1",
        "jeta_1",
        "jphi_1",
        "jpt_2",
        "jeta_2",
        "jphi_2",
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
    print(df)
    import pdb; pdb.set_trace()
    df["predictions_{}".format(NNname)] = loaded_model.predict(df[inputs])
    print(df)
    tree_dtype = {}
    tree_data = {}
    for b in [k for k in df.keys() if 'predictions' in k]:
        tree_dtype[b] = df[b].dtype.name
        if tree_dtype[b] == 'uint64':
            tree_dtype[b] = 'int64'
        tree_data[b] = np.array(df[b])

    if not arg.dry:
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
