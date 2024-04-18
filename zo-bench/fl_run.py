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
from federated_learning.our_framework import ourFramework
from federated_learning.fl_utils import *

os.environ["TRANSFORMERS_CACHE"] = "./cache"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AutoConfig.register("mistral", MistralConfig)
AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM)

def main():
    
    fl = False
    args,fed_args = parse_args()
    
    if hasattr(fed_args, "fed_alg") and fed_args.fed_alg != None:
        fl = True
        
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
    
    
    task = get_task(args.task_name)
    # fl
    if fl:
        partition_proportions = get_local_samples_distribution(fed_args) 

        if args.trainer == "zo_dp":
            framework = ourFramework(args, task, fed_args)
        else:
            framework = Framework(args, task, fed_args)

        # if has lastest round, load the model
        ch_round = 0
        if os.path.exists(os.path.join(args.output_dir, "round.txt")):
            with open(os.path.join(args.output_dir, "round.txt"), "r") as f:
                ch_round = int(f.read())
                if os.path.exists(os.path.join(args.output_dir, f"round_{ch_round}.pth")):
                    framework.model.load_state_dict(torch.load(os.path.join(args.output_dir, f"round_{ch_round}.pth")))
                    logging.info(f"Loading the model from round_{ch_round}.pth")
                else:
                    logging.info(f"round_{ch_round}.pth does not exist")
                    torch.save(framework.model.state_dict(), os.path.join(args.output_dir, "round_0.pth"))
        else:
            torch.save(framework.model.state_dict(), os.path.join(args.output_dir, "round_0.pth"))
        local_updates = None

        # fl
        for round in range(ch_round, fed_args.num_rounds):
            sampled_clients = select_clients(fed_args)
            print(f">> ==================== Round {round+1} : {sampled_clients} ====================")
            
            print(f"clients_this_round: {sampled_clients}")
            
            if args.trainer == "zo_dp":
                framework.before_broadcast()
                
            for i, client_id in enumerate(sampled_clients):
                # local train start
                weight = partition_proportions[client_id]
                train_client(args, framework, client_id, round, weight, logger, wandb)
                
                if args.trainer == "zo_dp":
                    framework.after_local_train(weight)
                    
                else: 
                    framework.weighted_update(weight, client_id)
                    if local_updates is None:
                        local_updates = framework.get_named_parameters_to_optm()
                    else:
                        local_updates += framework.get_named_parameters_to_optm()
                
                if i != len(sampled_clients) - 1:
                    framework.model.load_state_dict(torch.load(os.path.join(args.output_dir, f"round_{round}.pth"))) # reset the model parameters to the global model for next client

            # client training done for one round
            local_updates = framework.local_es_mangnitude_grads if args.trainer == "zo_dp" and fed_args.fed_alg == "fedavg" else local_updates
            global_aggregation(args, fed_args, framework, local_updates, round)

            # save the lastest round (int) 
            with open(os.path.join(args.output_dir, "round.txt"), "w") as f:
                    f.write(str(round+1))
            local_updates = None
            
            # global eval samples
            framework.fl_global_eval()

                
    else:
        # Initialize trainer and load model
        framework = Framework(args, task)
        framework.start_run()


if __name__ == "__main__":
    
    
    
    main()
