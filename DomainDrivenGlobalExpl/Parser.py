import argparse
import CONSTANTS

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default = 42,
                        help="Random Seed")
    parser.add_argument("--fold", type=int, default = 0,
                        help="Fold for cross validation")
    # Add argument for date_tag
    parser.add_argument('--date_tag', type=str, default='1225',
                        help='Tag representing the date of the experiment.')
    parser.add_argument('--output_dir', type=str, default='EXPT-01',
                        help='Directory where results are stored')

    # Add argument for dataset_name with choices
    parser.add_argument('--dataset_name', type=str, default='Mutagenicity',
                        choices=CONSTANTS.DATASET_COLUMN.keys(),
                        help='Name of the dataset to be used.')
    
    parser.add_argument("--column_name", type=str, default=None, help="Name to use of searching for column(defaults to dataset_name if not specified)")

    
    parser.add_argument('--task_type', type=str, default='BinaryClass',
                        choices=['BinaryClass', 'MultiTask','MultiClass', 'Regression'],
                        help='Type of prediction task.')
    
    parser.add_argument('--path', type=str, default='/nfs/stak/users/kokatea/hpc-share/ChemIntuit/MOSE-GNN/DICTIONARY',
                       help='Path to DICTOIONARY Folder used during vocabulary creation')
    
    parser.add_argument('--algorithm', type=str, default="None",
                        choices=["None","RBRICS", "MGSSL","PRESERVE_ALKANE_CARBONYL"],
                        help='Type of prediction task.')

    parser.add_argument("--base_importance", type=float, default = 0.0, help="Start for every motif parameter")
    parser.add_argument("--unk_importance", type=float, default = 1.0, help="Weightage given for rare unknown motifs")
    parser.add_argument(
        "--use_gumbel_softmax",
        dest="use_gumbel_softmax",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If False uses sigmoid activation",
    )
    
    parser.add_argument(
        "--use_annealing",
        dest="use_annealing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If False uses sigmoid activation without annealing",
    )
    parser.add_argument('--gumbel_tau', type=float, default=1.0,
                        help='Temperature parameter for annealing. Default is 1.0.')
    
    parser.add_argument(
        "--learn_unknown",
        dest="learn_unknown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If False uses fixed values for unknown motifs",
    )
    
    parser.add_argument(
        "--use_zero_weight",
        dest="use_zero_weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True uses zero weight for masked nodes",
    )

    parser.add_argument(
        "--use_stl",
        dest="use_stl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If False uses activation without binarization",
    )


    # Add arguments based on the config dictionary
    parser.add_argument('--num_mp_layers', type=int, default=2,
                        help='Number of message passing layers. Default is 2.')
    parser.add_argument('--layer_type', type=str, default='GINConv',
                        choices=['GIN', 'GCN', 'GAT', 'SAGE', 'PNA'],
                        help='Type of message passing. Default is GINConv.')
    parser.add_argument('--model_type', type=str, default='DualParam',
                        choices=['Vanilla', 'SingleParam', 'MultiChannel', 'SingleChannel'],
                        help='Type of message passing. Default is GINConv.')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units. Default is 16.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs for training. Default is [200].')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for training. Default is 0.0001.')
    parser.add_argument('--expl_lr', type=float, default=0.001,
                        help='Learning rate for training. Default is 0.001.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training. Default is 64.')
    parser.add_argument('--size_reg', type=float, default=0.0,
                        help='Size regularization parameter. Default is 0.0001.')
    # parser.add_argument('--class_reg', type=float, default=0.0,
    #                     help='Class regularization parameter. Default is 0.0.')
    parser.add_argument('--ent_reg', type=float, default=0.2,
                        help='Entropy regularization parameter. Default is 0.2.')
    
    parser.add_argument('--patience', type=int, default=10,
                        help='For Early Stopping. Default 10.')
    
    
    # Add argument for ignore_unknowns
    parser.add_argument('--ignore_unknowns', action='store_true', default=False,
                            help='Flag to ignore unknowns. Default is False. If set, will be True.')
    
    parser.add_argument('--vanilla_dir', type=str, default='EXPT-01',
                        help='Directory where vanilla model results are stored')

    args = parser.parse_args()
    
    return args

