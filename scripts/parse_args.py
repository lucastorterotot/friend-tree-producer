import argparse


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Script to manage condor batch system jobs for the executables and their outputs."
    )
    parser.add_argument(
        "--executable",
        required=True,
        choices=[
            "SVFit",
            "MELA",
            "NNScore",
            "NNMass",
            "NNrecoil",
            "FakeFactors",
            "ZPtMReweighting",
            "TauTriggers",
        ],
        help="Executable to be used for friend tree creation ob the batch system.",
    )
    parser.add_argument(
        "--batch_cluster",
        required=True,
        choices=["naf", "naf7", "etp6", "etp7", "lxplus6", "lxplus7", "rwth"],
        help="Batch system cluster to be used.",
    )
    parser.add_argument(
        "--command",
        required=True,
        choices=["submit", "collect", "check"],
        help="Command to be done by the job manager.",
    )
    parser.add_argument(
        "--input_ntuples_directory",
        required=True,
        help="Directory where the input files can be found. The file structure in the directory should match */*.root wildcard.",
    )
    parser.add_argument(
        "--shadow_input_ntuples_directory",
        type=str,
        help="Directory where the input files can be found. The file structure in the directory should match */*.root wildcard.",
    )
    parser.add_argument(
        "--events_per_job",
        required=True,
        type=int,
        help="Event to be processed by each job",
    )
    parser.add_argument(
        "--cmssw_tarball",
        required=False,
        help="Path to the tarball of this CMSSW working setup.",
    )
    parser.add_argument(
        "--friend_ntuples_directories",
        nargs="+",
        default=[],
        help="Directory where the friend files can be found. The file structure in the directory should match the one of the base ntuples. Channel dependent parts of the path can be inserted like /commonpath/{et:et_folder, mt:mt_folder,tt:tt_folder}/commonpath.",
    )
    parser.add_argument(
        "--walltime",
        default=-1,
        type=int,
        help="Walltime to be set for the job (in seconds). If negative, then it will not be set. [Default: %(default)s]",
    )
    parser.add_argument(
        "--cores",
        default=1,
        type=int,
        help="Number of cores to be used for the collect command. [Default: %(default)s]",
    )
    parser.add_argument(
        "--max_jobs_per_batch",
        default=10000,
        type=int,
        help="Maximal number of job per batch. [Default: %(default)s]",
    )
    parser.add_argument(
        "--mode",
        default="local",
        choices=["local", "xrootd"],
        type=str,
        help="Definition of file access",
    )
    parser.add_argument(
        "--input_server",
        default="GridKA",
        choices=["GridKA", "DESY", "EOS", "RWTH"],
        type=str,
        help="Definition of server for inputs",
    )
    parser.add_argument(
        "--output_server_xrootd",
        default="GridKA",
        choices=["GridKA", "DESY", "RWTH"],
        type=str,
        help="Definition of xrootd server for ouputs",
    )
    parser.add_argument(
        "--se-path", dest="se_path", default=None, type=str, help="se path for outputs"
    )
    parser.add_argument(
        "--extended_file_access",
        default=None,
        type=str,
        help="Additional prefix for the file access, e.g. via xrootd.",
    )
    parser.add_argument(
        "--custom_workdir_path",
        default=None,
        type=str,
        help="Absolute path to a workdir directory different from $CMSSW_BASE/src.",
    )
    parser.add_argument(
        "--restrict_to_channels",
        nargs="+",
        default=["em","mt","et","tt"],
        help="Produce friends only for certain channels",
    )
    parser.add_argument(
        "--restrict_to_shifts",
        nargs="+",
        default=[],
        help="Produce friends only for certain shifts",
    )
    parser.add_argument(
        "--restrict_to_samples_wildcards",
        nargs="+",
        default=[],
        help="Produce friends only for samples matching the path wildcard",
    )
    parser.add_argument(
        "--conditional",
        type=bool,
        default=False,
        help="Use conditional network for all eras or single era networks.",
    )
    parser.add_argument("--dry", action="store_true", default=False, help="dry run")
    parser.add_argument("--debug", action="store_true", default=False, help="debug")
    parser.add_argument(
        "--extra-parameters",
        type=str,
        default="",
        help="Extra parameters to be appended for each call",
    )
    parser.add_argument("--all", action="store_true", default=False, help="debug")
    return parser
