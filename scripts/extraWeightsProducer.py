#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Example:
        python HiggsAnalysis/friend-tree-producer/scripts/extraWeightsProducer.py \
            --input /nfs/dust/cms/group/higgs-kit/Legacy/MSSM/mva_v3/merged/2016/SUSYGluGluToHToTauTauM140_RunIISummer16MiniAODv3_PUMoriond17_13TeV_MINIAOD_pythia8_v2/SUSYGluGluToHToTauTauM140_RunIISummer16MiniAODv3_PUMoriond17_13TeV_MINIAOD_pythia8_v2.root \
            --output-dir  /nfs/dust/cms/user/glusheno/afs/RWTH/KIT/FriendTreeProducer/CMSSW_10_2_14/src/outputs_extraWeightsProducer \
            --start 0 --end 11 \
            --all-relevant-pipelines \
            --dry
'''
import re
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
        description="Calculate fake factors and create friend trees.")
    parser.add_argument("--input", required=True, type=str, help="Input file.")

    parser.add_argument("--tree", default="ntuple", type=str, help="Name of the root tree.")
    parser.add_argument("--enable-logging", action="store_true", help="Enable loggging for debug purposes.")
    parser.add_argument("--cmsswbase", default=os.environ['CMSSW_BASE'], help="Set path for to local cmssw for submission with Grid-Control")

    parser.add_argument("--first-entry", "--first_entry", "--start", default=0, type=int, help="Index of first event to process.")
    parser.add_argument("--last-entry", "--last_entry", "--end", default=-1, type=int, help="Index of last event to process.")

    parser.add_argument("--pipeline", "--pipelines", "--folder", nargs="?", default=None, type=str, help="Directory within rootfile.")
    parser.add_argument("--output-dir", type=str, default='.', help="Tag of output files.")

    parser.add_argument("--config", nargs="?", type=str, default=None, help="Config")

    parser.add_argument('--from-histograms', action='store_true', default=False, help='determine fractions')
    parser.add_argument('--dry', action='store_true', default=False, help='dry run')
    parser.add_argument('--all-relevant-pipelines', action='store_true', default=False, help='dry run')

    parser.add_argument('-w', type=str, default="somenewweight", help='output branch waight name')
    parser.add_argument('-s', type=str, help='x,y,z.. for reweighting')
    return parser.parse_args()


def apply_with_hist():
    pass


em_pipelines = [
    "nominal",
    "tauEsOneProngUp",
    "tauEsThreeProngDown",
    "tauEsOneProngDown",
    "tauEsOneProngOnePiZeroDown",
    "tauEsThreeProngUp",
    "tauEsOneProngOnePiZeroUp",
    "metRecoilResponseUp",
    "metRecoilResponseDown",
    "metRecoilResolutionDown",
    "metRecoilResolutionUp",
    "metUnclusteredEnUp",
    "metUnclusteredEnDown",
    "btagEffUp",
    "btagMistagDown",
    "btagMistagUp",
    "btagEffDown",
    "jecUncEta3to5Down",
    "jecUncRelativeBalUp",
    "jecUncRelativeBalDown",
    "jecUncEta0to5Down",
    "jecUncEC2Down",
    "jecUncEta0to5Up",
    "jecUncEta0to3Up",
    "jecUncRelativeSampleUp",
    "jecUncEta0to3Down",
    "jecUncEC2Up",
    "jecUncRelativeSampleDown",
    "jecUncEta3to5Up",
    "eleScaleUp",
    "eleSmearDown",
    "eleScaleDown",
    "eleSmearUp",
]


def filter_data_channels(channels, datafile):
    if "SingleElectron" in datafile or "_ElTau" in datafile:
        channels = set(channels) & set(["et"])
    elif "SingleMuon" in datafile or "_MuTau" in datafile:
        channels = set(channels) & set(["mt"])
    elif "MuonEG" in datafile or "_ElMu" in datafile:
        channels = set(channels) & set(["em"])
    elif ("Tau" in datafile and "Run200" in datafile) or "_TauTau" in datafile:
        channels = set(channels) & set(["tt"])
    else:
        channels = set(channels) & set(["em", "et", "mt", "tt"])


def addBranch(
        datafile,
        era,

        outputfile,
        configpath,
        treename='ntuple',
        rootfilemode='read',
        eventrange=[0, -1],
        dry=False,
        channels=["em"],
        pipelines=em_pipelines,
        all_relevant_pipelines=False,
):

    # HiggsAnalysis/friend-tree-producer/data/config_extraweights.yaml
    config = yaml.load(open(configpath))

    rootfilemode = rootfilemode.lower()
    if rootfilemode not in ["update", "recreate"] and not dry:
        raise Exception("Mode %s not appropriate for create of ROOT file. Please choose from 'update' and 'recreate'" % rootfilemode)

    # Prepare output
    if dry:
        print "...Would initialize output file %s" % outputfile
    else:
        print "...initialize output file %s" % outputfile
        if not os.path.exists(os.path.dirname(outputfile)):
            os.makedirs(os.path.dirname(outputfile))
        output_file = ROOT.TFile(outputfile, rootfilemode)

    filter_data_channels(channels, datafile)

    # Prepare data inputs
    input_file = ROOT.TFile(datafile, "READ")

    for channel in channels:
        if all_relevant_pipelines:
            pipelines = []
            for key in input_file.GetListOfKeys():
                if key.GetName().startswith(channel):
                    pipelines.append(key.GetName().split('/')[0].replace(channel + '_', ''))
            print 'process pipelines: %s' % pipelines

        for pipeline in pipelines:
            pipeline = pipeline.replace(channel + '_', '')

            fullPipeline = "%s_%s" % (channel, pipeline)
            if fullPipeline not in config.keys():
                print "SKIPPING PIPELINE NOT FOUND IN CONFIGS: %s" % (fullPipeline)
                continue

            # Prepare data inputs
            input_tree = input_file.Get("%s_%s/%s" % (channel, pipeline, treename))

            # Prepare dir
            if dry:
                print 'Would prepare root dir:  "%s_%s" / %s ' % (channel, pipeline, treename)
            else:
                output_root_dir = output_file.mkdir("%s_%s" % (channel, pipeline))
                output_root_dir.cd()
                output_tree = ROOT.TTree(treename, treename)

            # disable unnecessary branches
            input_tree.SetBranchStatus("*", 0)
            # Prepare branches
            output_buffer = {}
            output_expr = {}
            variables = set()
            for branch in config["%s_%s" % (channel, pipeline)].keys():
                output_buffer[branch] = numpy.zeros(1, dtype=float)
                if not dry:
                    output_tree.Branch(branch, output_buffer[branch], "%s/%s" % (branch, config["%s_%s" % (channel, pipeline)][branch]['type']))
                    output_expr[branch] = config["%s_%s" % (channel, pipeline)][branch]['formula']
                    vv = set(re.findall(r'[a-zA-Z_]\w*', output_expr[branch])) - set(['abs', 'min', 'max'])
                    for r in vv:
                        variables.add(r)
                        print "active Branch Name split:", r
                        input_tree.SetBranchStatus(r, 1)

            # Fill tree
            if eventrange[1] > 0:
                nev = eventrange[1] - eventrange[0] + 1
                if eventrange[1] >= input_tree.GetEntries():
                    raise Exception("The last entry exceeds maximum")
            else:
                nev = input_tree.GetEntries() - eventrange[0]

            printout_on = [eventrange[0] + i * int(nev / 10) for i in range(0, 11)]
            for evt_i, event in enumerate(input_tree):
                if eventrange is not None:
                    if evt_i < eventrange[0]:
                        continue
                    elif evt_i > eventrange[1] and eventrange[1] >= 0:  # latter condition allows to set negative upper limit in order to have it ignored
                        break
                    elif evt_i in printout_on:
                        print "\t ...processing %d %% [evt_i %d]" % (printout_on.index(evt_i) * 10, evt_i)

                # Evaluating weights
                for branch_name in output_buffer.keys():
                    s = copy.deepcopy(output_expr[branch])
                    s = s.replace('&&', 'and').replace('||', 'or').replace('!', 'not ')

                    ddict = {
                        'min': min,
                        'max': max,
                        'abs': abs,
                    }
                    for v in variables:
                        ddict[v] = getattr(event, v)
                    varvalue = eval(s, ddict)

                    output_buffer[branch_name] = varvalue

                if not dry:
                    output_tree.Fill()

            # Save
            if not dry:
                output_tree.Write()
                print "Successfully output_tree: %s / %s" % (outputfile, output_tree)

        print 'Done pipeline: %s' % pipeline
    print "Successfully finished file: %s " % (outputfile)

    # Clean up
    input_file.Close()
    if not dry:
        output_file.Close()
    return 0


def main(args):

    nickname = os.path.basename(args.input).replace(".root", "")

    # Get path to cmssw and fakefactordatabase
    cmsswbase = args.cmsswbase

    # Determine era
    with open(cmsswbase + "/src/HiggsAnalysis/friend-tree-producer/data/input_params/datasets.json") as json_file:
        datasets = json.load(json_file)
    era = str(datasets[nickname]["year"])

    # Create friend tree
    output_path = os.path.join(args.output_dir, nickname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    addBranch(
        datafile=args.input,
        era=era,
        eventrange=[args.first_entry, args.last_entry],
        outputfile=os.path.join(output_path, '_'.join(filter(None, [nickname, args.pipeline, str(args.first_entry), str(args.last_entry)])) + '.root'),
        configpath=args.config if args.config is not None else cmsswbase + "/src/HiggsAnalysis/friend-tree-producer/data/config_extraWeightsProducer.yaml",
        treename=args.tree,
        rootfilemode="recreate" if not args.dry else 'read',
        dry=args.dry,
        pipelines=em_pipelines if args.pipeline is None else [args.pipeline] if isinstance(args.pipeline, six.string_types) else args.pipeline,
        all_relevant_pipelines=args.all_relevant_pipelines,
    )

    print 'done'


if __name__ == "__main__":
    args = parse_arguments()

    if args.enable_logging:
        setup_logging(
            "extraWeightsProducer%s_%s_%s_%s.log" % (
                os.path.basename(args.input).replace(".root", ""),
                args.folder,
                args.first_entry,
                args.last_entry
            ),
            logging.WARNING
        )

    main(args)
