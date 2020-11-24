#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Example:
        python HiggsAnalysis/friend-tree-producer/scripts/add_ML_models_prediction_in_root_file.py \
            --input <path to root file> \
            --DNNs <path to DNN json files> \
            --XGBs <path to XGB json files> \
            --output-dir  <out dir> \
            --dry
"""

import os
import argparse
import logging
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

    parser.add_argument("--DNNs", required=False, type=str, help="Input DNN json files.", default="")

    parser.add_argument("--XGBs", required=False, type=str, help="Input XGB json files.", default="")

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

def main(args):
    print(args)

    DNN_jsons = args.DNNs.split(',')
    if len(DNN_jsons) > 0:
        command = "add_DNN_model_prediction_in_root_file.py"
        command += " --input={}".format(args.input)
        command += " --DNNs={}".format(args.DNNs)
        command += " --tree={}".format(args.tree)
        command += " --channels={}".format(args.channels)
        command += " --categories={}".format(args.categories)
        if args.enable_logging:
            command += " --enable-logging"
        if args.first_entry != 0:
            command += " --first_entry={}".format(args.first_entry)
        if args.last_entry != -1:
            command += " --last_entry={}".format(args.last_entry)
        if args.pipeline != None:
            command += " --pipeline={}".format(args.pipeline)
        command += " --output-dir={}".format(args.output_dir)
        os.system(command)
    XGB_jsons = args.XGBs.split(',')    
    if len(XGB_jsons) > 0:
        command = "add_XGB_model_prediction_in_root_file.py"
        command += " --input={}".format(args.input)
        command += " --XGBs={}".format(args.XGBs)
        command += " --tree={}".format(args.tree)
        command += " --channels={}".format(args.channels)
        command += " --categories={}".format(args.categories)
        if args.enable_logging:
            command += " --enable-logging"
        if args.first_entry != 0:
            command += " --first_entry={}".format(args.first_entry)
        if args.last_entry != -1:
            command += " --last_entry={}".format(args.last_entry)
        if args.pipeline != None:
            command += " --pipeline={}".format(args.pipeline)
        command += " --output-dir={}".format(args.output_dir)
        os.system(command)

if __name__ == "__main__":
    args = parse_arguments()

    if args.enable_logging:
        setup_logging(
            "add_ML_model_prediction_in_root_file_%s_%s_%s_%s.log"
            % (
                os.path.basename(args.input).replace(".root", ""),
                args.folder,
                args.first_entry,
                args.last_entry,
            ),
            logging.WARNING,
        )

    main(args)
