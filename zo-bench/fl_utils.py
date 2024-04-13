
import numpy as np
from utils import *


def get_sampler(fed_args, weights=None):
    num_clients = fed_args.num_clients
    if weights is None:
        return [1/num_clients for _ in range(num_clients)]
    else:
        return weights

def select_clients(fed_args, weights=None):
    if weights is None:
        get_sampler(fed_args)
    num_clients = fed_args.num_clients
    sample_clients = fed_args.sample_clients
    return random.choices(range(num_clients), k=sample_clients, weights=weights)
   

def global_aggregation(fed_args, framework, local_dict_list, sampled_clients, num_samples, total_samples, round):
    if fed_args.fed_alg == 'fedavg':
        model_dict = framework.get_named_parameters_to_optm()
        for key in model_dict.keys():
            model_dict[key] = sum([local_dict_list[client_id][key] * num_samples[client_id] / total_samples for client_id in sampled_clients])
        framework.set_model_dict(model_dict)
        logging.info(f"Round {round+1} global aggregation done.")
        return model_dict
    else:
        raise NotImplementedError(f"Aggregation method {fed_args.fed_alg} is not implemented.")
    
def train_client(args, framework, client_id, current_round, local_dataset, local_trainset, logger, wandb):
    print(f"Training client {client_id}")
    # Initialize trainer and load model
    # task = local_datasets[i]
    framework.task = local_dataset

    # ZO-Bench Added
    # We add these parameters to evaluate the model during the training.
    # These two parameters will be used in the training loop
    # args.task = task
    # args.framework = framework

    if args.train_set_seed is not None or args.num_train_sets is not None:

        # Training goes to this way

        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(local_trainset):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = local_dataset.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = local_dataset.valid_samples

            if args.trainer != "none":
                # Here the training samples are seperated
                if args.num_dev is not None:
                    logging.info(f"args.num_dev is not None: {args.num_dev}") 
                    # Dev samples
                    # assert args.num_dev + args.num_train <= len(train_samples), f"num_dev({args.num_dev})+num_train({args.num_train}) is more than actual num of training samples ({len(train_samples)})."
                    dev_samples = train_samples[-args.num_dev:]
                    train_samples = train_samples[:-args.num_dev]
                    logger.info("Dev samples: %d" % len(dev_samples))
                    logger.info("Train samples: %d" % len(train_samples))
                else:
                    dev_samples = None
                    logger.info("Train samples: %d" % len(train_samples))
                    logger.info("No dev samples")

                args.dev_samples = dev_samples
                args.eval_samples = eval_samples

                # Training
                framework.train(client_id, current_round, train_samples, dev_samples if dev_samples is not None else eval_samples, eval_samples)

                if not args.no_eval:  # This is True
                    metrics = framework.evaluate([], eval_samples, description="Evaluating on the Test Set")
                    _keys = list(metrics.keys())
                    for m in _keys:
                        metrics["test_" + m] = metrics[m]
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate(
                            [], dev_samples, description="Evaluating on the Validation Set"
                        )
                        _keys = list(dev_metrics.keys())
                        for m in _keys:
                            metrics["val_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)
            logger.info(metrics)
            wandb.log(metrics)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                wandb.log(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" + result_file_tag(
                        args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)
            if args.trainer != "none" and args.clean_model_at_end:
                framework.delete_checkpoints()

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = local_dataset.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = local_dataset.valid_samples
        metrics = framework.evaluate(local_trainset, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        wandb.log(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(
                args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)
    