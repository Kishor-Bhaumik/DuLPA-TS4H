import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model_pct_blup import MODEL 
from arguments import  get_args
from dataloader import get_dataset
from data_process import  opp_data,organize_file , sbhar_data, wisdm_data
import os
import numpy as np
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import wandb
import warnings
warnings.filterwarnings("ignore")

def model_output(args):
    all_users_test_f1_scores=[]
    all_users_test_auroc_scores=[]
    num_domains = args.N_users
    
    # Store results for file output
    results_log = []
    
    if args.use_group_testing:
        path =  args.dataset_path + 'processed_data/group_1'
        if not os.path.exists(path):
            print("Organizing files for group testing...")
            organize_file.organize_files(args.dataset_path, args.subject_groups)
        args.N_users = len(args.subject_groups)*len(args.subject_groups[0])-len(args.subject_groups[0])+1
        num_domains= len(args.subject_groups)
    count = 0

    for i in range(num_domains):
        
        expname= f'{args.exp_name}_{i}'
        wandb_logger = None
        if args.logger:
            wandb_logger = WandbLogger(project=args.project_name ,
                                name=expname,
                                group=args.experiment_group,
                                notes=args.notes,
                                job_type=args.model_variant,
                                tags= ["HAR", args.experiment_group, args.model_variant],
                                log_model=False)
        args.test_subject = i if not args.use_group_testing else count 
        print("\n\n")
        print(f"Evaluating on subject {args.test_subject+1}...")
        
        # Log this to results for file
        results_log.append(f"\n\nEvaluating on subject {args.test_subject+1}...")
        
        if args.use_group_testing: count+= len(args.subject_groups[i])

        train_loader_S, train_loader_T, test_loader = get_dataset(args.batch_size, args.test_subject, args)
        
        train_loader = {"source": train_loader_S, "target": train_loader_T}

        model = MODEL(args)
        
        # Suppress PyTorch Lightning's verbose output for file logging
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            logger=wandb_logger if args.logger else None, 
            accelerator='gpu',
            devices=[args.device_id],
            # limit_train_batches=1, 
            # limit_test_batches=1,
            enable_progress_bar=True,  # Keep progress bar for terminal
            enable_model_summary=False,  # Reduce verbosity
        ) 
        
        trainer.fit(model, train_dataloaders=train_loader)
        test_results = trainer.test(model, test_loader)
        
        result_msg = f" 🟥🟥🟥🟥🟥 That was the results for user (or domain or subject) : {i+1}  🟥🟥🟥🟥🟥"
        print(result_msg)
        results_log.append(result_msg)

        test_f1_macro = test_results[0].get('test_f1_macro', 0.0)
        test_auroc = test_results[0].get('test_auroc_macro', 0.0)
        all_users_test_f1_scores.append(test_f1_macro)
        all_users_test_auroc_scores.append(test_auroc)
        
        # Log individual results
        results_log.append(f"Subject {i+1} - F1: {test_f1_macro:.4f}, AUROC: {test_auroc:.4f}")
        
        if args.logger:
            wandb_logger.experiment.summary[f"test_f1_user_{i}"] = test_f1_macro
            wandb_logger.experiment.summary[f"test_auroc_user_{i}"] = test_auroc
            wandb_logger.experiment.finish()
            
    std_f1 = np.std(all_users_test_f1_scores)
    std_auroc = np.std(all_users_test_auroc_scores)
    
    avg_f1 = sum(all_users_test_f1_scores)/len(all_users_test_f1_scores)
    avg_auroc = sum(all_users_test_auroc_scores)/len(all_users_test_auroc_scores)
    
    # Prepare final results for both terminal and file
    final_results = []
    final_results.append("\n\n\n\n")
    final_results.append("********** Cross Validation average Result is below **********")
    final_results.append("\n")
    colors = ["🟥", "🟧", "🟨", "🟩", "🟦", "🟪"]
    final_results.append("".join(color * 8 for color in colors))
    final_results.append(f"╔════════════════════════════════════════════════════════════════════════╗")
    final_results.append(f"║ Average F1: {avg_f1:.4f} Average auroc: {avg_auroc:.4f} ║")
    final_results.append(f"╚════════════════════════════════════════════════════════════════════════╝")
    final_results.append(f"╔════════════════════════════════════════════════════════════════════════╗")
    final_results.append(f"║ AUROC std: {std_auroc:.4f} macro F1 std: {std_f1:.4f} ║")
    final_results.append(f"╚════════════════════════════════════════════════════════════════════════╝")
    
    # Print to terminal
    for line in final_results:
        print(line)
    
    # Return results and logs for file writing
    return avg_f1, avg_auroc, std_auroc, std_f1, results_log + final_results

def main(config=None):
    args = get_args()

    with wandb.init(config=config) as run:
        config = wandb.config
        #print(f"Running with config: {config}")
        for key, value in config.items():
            setattr(args, key, value)

        path = args.dataset_path + 'processed_data/sub0_features.npy'

        if args.dataset == 'opp':
            if not os.path.exists(path):
                opp_data.preprocess_opportunity(args.dataset_path , args.window, args.overlap)
        
        elif args.dataset == 'sbhar':
            if not os.path.exists(path):
                sbhar_data.preprocess_sbhar(args.dataset_path)
        
        elif args.dataset == 'wisdm':
            if not os.path.exists(path):
                wisdm_data.preprocess_wisdm(args.dataset_path, args.window, args.overlap)
        
        else:
            raise ValueError("Invalid dataset name. Please choose 'opp' or 'sbhar' or 'wisdm'")

        pl.seed_everything(args.seed)
        
        # Run the model and get results + logs
        test_f1_score, test_auroc, std_au, std_f1, file_logs = model_output(args)
        run.log({"test_f1": test_f1_score, "test_auroc": test_auroc})
        run.summary["test_f1"] = test_f1_score
        run.summary["test_auroc"] = test_auroc




if __name__ == '__main__':

    args = get_args()
    
    path = args.dataset_path + 'processed_data/sub0_features.npy'

    if args.dataset == 'opp':
        if not os.path.exists(path):
            opp_data.preprocess_opportunity(args.dataset_path , args.window, args.overlap)
    
    elif args.dataset == 'sbhar':
        if not os.path.exists(path):
            sbhar_data.preprocess_sbhar(args.dataset_path)
    
    elif args.dataset == 'wisdm':
        if not os.path.exists(path):
            wisdm_data.preprocess_wisdm(args.dataset_path, args.window, args.overlap)
    
    else:
        raise ValueError("Invalid dataset name. Please choose 'opp' or 'sbhar' or 'wisdm'")

    pl.seed_everything(args.seed, workers=True)
    
    # Run the model and get results + logs
    test_f1_score, test_auroc, std_au, std_f1, file_logs = model_output(args)
    
    # Write only the essential results to file
    with open(args.dataset+'_result_output.txt', 'w') as f:
        for line in file_logs:
            f.write(line + '\n')
