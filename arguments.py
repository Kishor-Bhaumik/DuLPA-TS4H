import argparse
import sys


def common_args():
    """Common arguments shared across all datasets"""
    parser = argparse.ArgumentParser(add_help=False)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--alpha', type=float, default=1, help='alpha')
    parser.add_argument('--test_subject', type=int, default=1, help='the subject to be tested')
    parser.add_argument('--scheduler', type=bool, default=True, help='scheduler')
    parser.add_argument('--scheduler_type', type=str, default='step', help='scheduler_type')
    parser.add_argument('--domain_feature_dim', type=int, default=128, help='domain_feature_dim')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout')
    
    # Loss weights
    parser.add_argument('--unsup_loss_weight', type=float, default=0.13, help='unsup_weight_loss_')
    parser.add_argument('--supervised_loss_weight', type=float, default=0.93, help='supervised_weight_loss_')
    # add beta
    parser.add_argument('--beta', type=float, default=0, help='beta for prototype loss')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=5, help='seed')
    parser.add_argument('--experiment_group', type=str, default='DuLPAWSDM', help='experiment type')
    parser.add_argument('--model_variant', type=str, default='1D_CNN', help='model variant')
    parser.add_argument('--entropy_temperature', type=float, default=0.5, help='entropy_temperature')
    parser.add_argument('--decay', type=float, default=0.3, help='decay for semantic loss centroid')
    parser.add_argument('--notes', type=str, default=' running for opp dataset', help='notes')
    parser.add_argument('--use_proto', type=bool, default=True, help='alpha')
    parser.add_argument('--logger', type=bool, default=False, help='wandb logging')
    parser.add_argument('--proto_update_freq', type=int, default=100, help='frequency of global prototype updates (in batches)')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID')

    # Prototype parameters
    parser.add_argument('--attraction_weight', type=float, default=0.5, help='attraction_weight')
    parser.add_argument('--repulsion_weight', type=float, default=0.3, help='repulsion_weight')
    parser.add_argument('--prior_strength', type=float, default=0.6, help='prior_strength')
    parser.add_argument('--project_name', type=str, default='DuLPA', help='project name for wandb')
    #s_par
    parser.add_argument('--s_par', type=float, default=0.5, help='source proportion in loss calculation')
    parser.add_argument('--momentum', type=float, default=0.4, help='blue momentum')
   
    
    return parser


def get_dataset_specific_args(dataset_name):
    """Get dataset-specific arguments based on dataset name"""
    
    if dataset_name.lower()== 'wisdm':
        return {
            'window': 100,
            'overlap': 50,
            'dataset_path': 'wisdm_data/',
            'dataset': 'wisdm',
            'N_users': 36,
            'num_classes': 6,
            'num_channel': 3,
            'use_group_testing': True,
            'subject_groups': [[0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24, 25, 26], [27, 28, 29, 30, 31, 32, 33, 34, 35]],
            'epochs': 25,
            'lr': 5e-4,
            'optimizer': 'sgd',
            'scheduler_step_size': 5,
            'scheduler_gamma': 0.95,
            'exp_name': 'wisdm_experiment'
        }
    
    elif dataset_name.lower() == 'sbhar':
        return {
            'window': 128,
            'overlap': 64,
            'dataset_path': 'sbhar_data/',
            'dataset': 'sbhar',
            'N_users': 30,
            'num_classes': 6,
            'num_channel': 9,
            'use_group_testing': True,
            'subject_groups': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]],
            'epochs': 40,
            'lr': 5e-5,
            'optimizer': 'sgd',
            'scheduler_step_size': 10,
            'scheduler_gamma': 0.95,
            'exp_name': 'sbhar_experiment'
        }
    
    elif dataset_name.lower()== 'opp':
        return {
            'window': 90,
            'overlap': 45,
            'dataset_path': 'opp_data/',
            'dataset': 'OPPORTUNITY',
            'N_users': 4,
            'num_classes': 5,
            'num_channel': 77,
            'use_group_testing': False,
            'subject_groups': None,  # Not used for opportunity
            'epochs': 10,
            'lr': 0.001,
            'optimizer': 'adam',
            'scheduler_step_size': 2,
            'scheduler_gamma': 0.1,
            'exp_name': 'opp_experiment'
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: wisdm, uschad, sbhar, opp, opportunity")


def create_unified_parser():
    """Create a unified parser that loads dataset-specific arguments"""
    parser = argparse.ArgumentParser(description='Human Activity Recognition with Multiple Datasets', parents=[common_args()])
    
    # Add dataset selection argument
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['wisdm', 'sbhar', 'opp'],
                       help='Dataset to use: wisdm, sbhar, opp')
    
    return parser


def get_args():
    """Main function to get arguments based on dataset"""
    # First parse to get dataset name
    parser = create_unified_parser()
    
    # Parse known args first to get dataset
    args, remaining = parser.parse_known_args()
    dataset_name = args.dataset
    
    # Get dataset-specific defaults
    dataset_defaults = get_dataset_specific_args(dataset_name)
    
    # Create new parser with dataset-specific arguments
    final_parser = argparse.ArgumentParser(description=f'HAR Training for {dataset_name.upper()}', parents=[common_args()])
    
    # Add dataset argument (already parsed)
    final_parser.add_argument('--dataset', type=str, default=dataset_name,
                             help='Dataset name')
    
    # Add dataset-specific arguments
    final_parser.add_argument('--window', type=int, default=dataset_defaults['window'], 
                             help=f'Time window length for {dataset_name}')
    final_parser.add_argument('--overlap', type=int, default=dataset_defaults['overlap'], 
                             help=f'Overlap between windows for {dataset_name}')
    final_parser.add_argument('--dataset_path', type=str, default=dataset_defaults['dataset_path'], 
                             help='Path to dataset')
    final_parser.add_argument('--N_users', type=int, default=dataset_defaults['N_users'], 
                             help='Number of users/domains')
    final_parser.add_argument('--num_classes', type=int, default=dataset_defaults['num_classes'], 
                             help='Number of classes')
    final_parser.add_argument('--num_channel', type=int, default=dataset_defaults['num_channel'], 
                             help='Number of channels')
    final_parser.add_argument('--use_group_testing', type=bool, default=dataset_defaults['use_group_testing'], 
                             help='Use group testing')
    
    # Add subject_groups only if it exists for this dataset
    if dataset_defaults['subject_groups'] is not None:
        final_parser.add_argument('--subject_groups', type=list, default=dataset_defaults['subject_groups'], 
                                 help='Subject groups for group adaptation')
    
    # Training parameters
    final_parser.add_argument('--epochs', type=int, default=dataset_defaults['epochs'], 
                             help='Number of training epochs')
    final_parser.add_argument('--lr', type=float, default=dataset_defaults['lr'], 
                             help='Learning rate')
    final_parser.add_argument('--optimizer', type=str, default=dataset_defaults['optimizer'], 
                             help='Optimizer')
    final_parser.add_argument('--scheduler_step_size', type=int, default=dataset_defaults['scheduler_step_size'], 
                             help='Scheduler step size')
    final_parser.add_argument('--scheduler_gamma', type=float, default=dataset_defaults['scheduler_gamma'], 
                             help='Scheduler gamma')
    # Experiment parameters
    final_parser.add_argument('--exp_name', type=str, default=dataset_defaults['exp_name'], 
                             help='Experiment name')
    
    # Parse all arguments
    return final_parser.parse_args()


# Usage example
if __name__ == "__main__":
    args = get_args()
    print(f"Running {args.dataset} with following config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Window size: {args.window}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Project: {args.project_name}")
