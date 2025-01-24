import argparse

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default = 42,
                        help="Random Seed")
    parser.add_argument("--fold", type=int, default = 0,
                        help="Fold for cross validation")
    # Add argument for date_tag
    parser.add_argument('--date_tag', type=str, default='0828',
                        help='Tag representing the date of the experiment.')
    parser.add_argument('--output_dir', type=str, default='EXPT-01',
                        help='Directory where results are stored')

    # Add argument for dataset_name with choices
    parser.add_argument('--dataset_name', type=str, default='Mutagenicity',
                        choices=['Mutagenicity', 'hERG', 'BBBP', 'tox21', 'esol', 'freesolv', 'Lipophilicity'],
                        help='Name of the dataset to be used.')
    
    parser.add_argument('--task_type', type=str, default='BinaryClass',
                        choices=['BinaryClass', 'MultiTask','MultiClass', 'Regression'],
                        help='Type of prediction task.')
    
    parser.add_argument('--algorithm', type=str, default="None",
                        choices=["None","RBRICS", "MGSSL"],
                        help='Type of prediction task.')

    parser.add_argument("--base_importance", type=float, default = 0.0, help="Start for every motif parameter")

    # Add arguments based on the config dictionary
    parser.add_argument('--num_mp_layers', type=int, default=2,
                        help='Number of message passing layers. Default is 2.')
    parser.add_argument('--layer_type', type=str, default='GINConv',
                        choices=['GINConv', 'GCNConv', 'GATConv'],
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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training. Default is 64.')
    parser.add_argument('--size_reg', type=float, default=0.0,
                        help='Size regularization parameter. Default is 0.0001.')
    parser.add_argument('--class_reg', type=float, default=0.0,
                        help='Class regularization parameter. Default is 0.0.')
    parser.add_argument('--ent_reg', type=float, default=0.2,
                        help='Entropy regularization parameter. Default is 0.2.')
    # Add argument for ignore_unknowns
    parser.add_argument('--ignore_unknowns', action='store_true', default=False,
                            help='Flag to ignore unknowns. Default is False. If set, will be True.')

    args = parser.parse_args()
    
    return args

