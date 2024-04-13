import copy
import os

import wandb
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from modeling_mistral import (
    MistralForCausalLM,
    MistralConfig
)
from tasks import *
from utils import *

from fl_utils import *

os.environ["TRANSFORMERS_CACHE"] = "./cache"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AutoConfig.register("mistral", MistralConfig)
AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM)

def main():
    args,fed_args = parse_args()
    if args.prefix_tuning:
        args.mode = "prefix"
    elif args.lora:
        args.mode = "lora"
    elif args.prompt_tuning:
        args.mode = "prompt"
    else:
        args.mode = "ft"
    args.tag = f"{args.trainer}-{args.task_name}-{args.template_ver}-{args.model_name.split('/')[-1]}-OPTIM_{args.mode}-STEP{args.max_steps}-{args.optimizer}-LR{args.learning_rate}-{args.lr_scheduler_type}-ZOEPS{args.zo_eps}-Q{args.q}"
    args.tag = "momen" + args.tag if args.momentum > 0 else args.tag
    args.run_name = args.tag
    args.output_dir = f"result/{args.tag}"
    args.result_file = f"result/{args.tag}/results.json"
    os.makedirs(args.output_dir, exist_ok=True)
    args.logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(args.logging_dir, exist_ok=True)

    wandb.init(project='zo-bench', name=args.tag, config=args)

    set_seed(args.seed)
    # task = get_task(args.task_name)
    local_datasets, local_train_num_samples, local_val_num_samples = get_local_datasets(args.task_name, fed_args, args)
    
    logging.info(f"Number of train samples for each client: {local_train_num_samples}")
    logging.info(f"Number of val samples for each client: {local_val_num_samples}")
    local_trainsets = get_local_trainsets(local_datasets, args)
    
    framework = Framework(args, None)
    global_dict = copy.deepcopy(framework.get_named_parameters_to_optm())
    
    local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
    
    ### Test
    for round in range(fed_args.num_rounds):
        global_model_path = os.path.join(args.output_dir, f"global_model_{round}.pth")
        # save framework.get_named_parameters_to_optm() to global_model_path
        torch.save(global_dict, global_model_path)
        
        sampled_clients = select_clients(fed_args)
        print(f">> ==================== Round {round+1} : {sampled_clients} ====================")
        
        print(f"clients_this_round: {sampled_clients}")
        # bug: 
        num_samples = [local_train_num_samples[client_id] for client_id in sampled_clients]
        total_samples = sum(num_samples)
        for client_id in sampled_clients:
            train_client(args, framework, client_id, round, local_datasets[client_id], local_trainsets[client_id], logger, wandb)

            framework.set_model_dict(global_dict) # reset the model parameters to the global model for next client
            local_dict_list[client_id] = copy.deepcopy(framework.get_named_parameters_to_optm())

        global_dict = global_aggregation(fed_args, framework, local_dict_list, sampled_clients, num_samples, total_samples, round)
            
            


if __name__ == "__main__":
    main()
