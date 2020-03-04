#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    Example:
        python HiggsAnalysis/friend-tree-producer/scripts/emQCDWeightsProducer.py \
            --input /pnfs/desy.de/cms/tier2/store/user/ohlushch/MSSM/merged/2017/VBFHToTauTauM125_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_powheg-pythia8_v1/VBFHToTauTauM125_RunIIFall17MiniAODv2_PU2017_13TeV_MINIAOD_powheg-pythia8_v1.root \
            --output-dir  /nfs/dust/cms/user/glusheno/afs/RWTH/KIT/FriendTreeProducer/CMSSW_10_2_14/src/outputs_emQCDWeights \
            --rooworkspace-file /nfs/dust/cms/user/glusheno/htt_scalefactors_legacy_2017.root \
            --start 0 --end 11 \
            --all-relevant-pipelines \
            --dry
'''
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

    parser.add_argument("--pipeline", default=None, type=str, help="Directory within rootfile.")
    parser.add_argument("--output-dir", type=str, default='.', help="Tag of output files.")

    parser.add_argument("--rooworkspace-file", type=str, default=None, help="List of files with friend trees.")
    parser.add_argument("--config", nargs="?", type=str, default=None, help="Config")

    parser.add_argument('--from-histograms', action='store_true', default=False, help='determine fractions')
    parser.add_argument('--dry', action='store_true', default=False, help='dry run')
    parser.add_argument('--all-relevant-pipelines', action='store_true', default=False, help='dry run')

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


def apply_with_rooworkspace(
        datafile,
        era,
        rooworkspace_file_name,
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

    config = yaml.load(open(configpath))
    if "rooworkspace" not in config.keys():
        raise Exception("Config file %s has to contain key 'rooworkspace'!" % configpath)

    rootfilemode = rootfilemode.lower()
    if rootfilemode not in ["update", "recreate"] and not dry:
        raise Exception("Mode %s not appropriate for create of ROOT file. Please choose from 'update' and 'recreate'" % rootfilemode)

    map_arguments = {}
    if 'map_arguments' in config:
        map_arguments = config['map_arguments']

    QCDFactorWorkspace = config["rooworkspace"]
    workspace_object_names = config["rooworkspace"].keys()

    # Reading rooworkspace file
    print "rooworkspace_file_name: %s" % rooworkspace_file_name
    rooworkspace_file = ROOT.TFile(rooworkspace_file_name, 'read')
    # gSystem->AddIncludePath("-I$ROOFITSYS/include");
    m_workspace = rooworkspace_file.Get("w")
    m_functors = {}
    variables = set()
    for object_name in workspace_object_names:
        variables.update(config["rooworkspace"][object_name]["WorkspaceObjectArguments"])
        m_functors[config["rooworkspace"][object_name]['WorkspaceWeightNames']] = {
            # 'functor': m_workspace.function(object_name).functor(
            #     m_workspace.argSet(','.join(config["rooworkspace"][object_name]["WorkspaceObjectArguments"]))),
            'function': m_workspace.function(object_name),
            'argSet': m_workspace.argSet(','.join(config["rooworkspace"][object_name]["WorkspaceObjectArguments"])),
            'arguments': config["rooworkspace"][object_name]["WorkspaceObjectArguments"],
        }
    variables_dict = {}
    for v in variables:
        variables_dict[v] = 0.0

    # Prepare output
    if dry:
        print "...Would initialize output file %s" % outputfile
    else:
        print "...initialize output file %s" % outputfile
        if not os.path.exists(os.path.dirname(outputfile)):
            os.makedirs(os.path.dirname(outputfile))
        output_file = ROOT.TFile(outputfile, rootfilemode)

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

            # Prepare data inputs
            input_tree = input_file.Get("%s_%s/%s" % (channel, pipeline, treename))
            # TODO: disable all variables that will not be needed for application with SetBranchStatus("*", 0)

            # Prepare dir
            if dry:
                print 'Would prepare root dir:  "%s_%s" / %s ' % (channel, pipeline, treename)
            else:
                output_root_dir = output_file.mkdir("%s_%s" % (channel, pipeline))
                output_root_dir.cd()
                output_tree = ROOT.TTree(treename, treename)

            # Prepare branches
            output_buffer = {}
            for branch in m_functors.keys():
                output_buffer[branch] = numpy.zeros(1, dtype=float)
                if not dry:
                    output_tree.Branch(branch, output_buffer[branch], "%s/D" % branch)

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
                for ws_name in m_functors.keys():
                    parameters = []
                    for arg in m_functors[ws_name]['arguments']:

                        arg0 = arg
                        if arg in map_arguments:
                            arg0 = map_arguments[arg]
                        elif '_' in arg:
                            if arg.startswith('e_'):
                                arg0 = arg.replace('e_', '') + '_1'
                            elif arg.startswith('m_'):
                                arg0 = arg.replace('m_', '') + '_2'
                            else:
                                raise Exception('unclear unpacking variable %s' % arg)

                        parameters.append(getattr(event, arg0))
                        m_functors[ws_name]["argSet"].setRealValue(arg, getattr(event, arg0))

                    output_buffer[ws_name] = m_functors[ws_name]['function'].getVal(m_functors[ws_name]['argSet'])
                    # output_buffer[ws_name] = m_functors[ws_name]['functor'].eval(parameters)

                if not dry:
                    output_tree.Fill()

            # Save
            if not dry:
                output_tree.Write()
                print "Successfully output_tree: %s " % (output_tree)

        print 'Done pipeline: %s' % pipeline
    print "Successfully finished file: %s / %s" % (outputfile, output_tree)

    # Clean up
    rooworkspace_file.Close()
    input_file.Close()
    if not dry:
        output_file.Close()
    print 'done'
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

    if not args.from_histograms:
        if args.rooworkspace_file is None:
            raise Exception("Rooworkspace file expected but not given")

        apply_with_rooworkspace(
            datafile=args.input,
            era=era,
            eventrange=[args.first_entry, args.last_entry],
            rooworkspace_file_name=args.rooworkspace_file,
            outputfile=os.path.join(output_path, '_'.join(filter(None, [nickname, args.pipeline, str(args.first_entry), str(args.last_entry)])) + '.root'),
            configpath=args.config if args.config is not None else cmsswbase + "/src/HiggsAnalysis/friend-tree-producer/data/config_emQCDWeightsProducer.yaml",
            treename=args.tree,
            rootfilemode="recreate" if not args.dry else 'read',
            dry=args.dry,
            pipelines=em_pipelines if args.pipeline is None else args.pipeline,
            all_relevant_pipelines=args.all_relevant_pipelines,
        )
    else:
        raise Exception('Hist not implemented')


if __name__ == "__main__":
    args = parse_arguments()

    if args.enable_logging:
        setup_logging(
            "emQCDWeightsProducer%s_%s_%s_%s.log" % (
                os.path.basename(args.input).replace(".root", ""),
                args.folder,
                args.first_entry,
                args.last_entry
            ),
            logging.WARNING
        )

    main(args)
