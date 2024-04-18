import numpy as np
from utils import *

def get_local_samples_distribution(fed_args):
    proportions = []
    if fed_args.split_strategy == "iid":
        proportions = [1/fed_args.num_clients] * fed_args.num_clients
    elif fed_args.split_strategy == "noniid":
        proportions = np.random.dirichlet(np.repeat(fed_args.alpha, fed_args.num_clients)) 
    return proportions


def select_clients(fed_args):
    num_clients = fed_args.num_clients
    sample_clients = fed_args.sample_clients
    client_list = np.random.permutation(num_clients)[:sample_clients]
    client_list = client_list.astype(int).tolist()
    return client_list


def global_aggregation(
    args,
    fed_args,
    framework,
    local_updates,
    round,
    save_step=1,
):
    if fed_args.fed_alg == "fedavg":
        # framework.model.cpu()
        framework.agg_model_parameters(local_updates)
        # framework.model.cuda()
        logging.info(f"Round {round+1} global aggregation done.")

    else:
        raise NotImplementedError(
            f"Aggregation method {fed_args.fed_alg} is not implemented."
        )
    if round % save_step == 0:
        torch.save(
            framework.model.state_dict(),
            os.path.join(args.output_dir, f"round_{round+1}.pth"),
        )  # save the global model for next round
        logging.info(f"Gloabl model saved to round_{round+1}.pth")
        # remove previous round model
        if round > 0:
            os.remove(os.path.join(args.output_dir, f"round_{round+1-save_step}.pth"))
            logging.info(f"round_{round+1-save_step}.pth removed.")


def train_client(
    args,
    framework,
    client_id,
    current_round,
    weight,
    logger,
    wandb,
):
    print(f"Training client {client_id}")
    # Initialize trainer and load model
    metrics = None
    num_eval = int(args.num_eval * weight) if args.num_eval is not None else None
    num_train = int(args.num_train * weight)
    num_dev = int(args.num_dev * weight) if args.num_dev is not None else None

    if args.train_set_seed is not None or args.num_train_sets is not None:
        train_sets = framework.task.sample_train_sets(
            num_train,
            num_dev,
            num_eval,
            num_train_sets=args.num_train_sets,
            seed=args.train_set_seed,
        )

        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = (
                train_set_id if args.train_set_seed is None else args.train_set_seed
            )

            # Sample eval samples
            if num_eval is not None:
                eval_samples = framework.task.sample_subset(
                    data_split="valid", seed=train_set_seed, num=num_eval
                )
            else:
                eval_samples = framework.task.valid_samples

            if args.trainer != "none":
                # Here the training samples are seperated
                if num_dev is not None:
                    logging.info(f"num_dev is not None: {num_dev}")
                    # Dev samples
                    if num_dev + num_train > len(train_samples):
                        num_train = int(0.7 * len(train_samples)) + 1
                        num_dev = len(train_samples) - num_train
                    dev_samples = train_samples[num_train : num_train + num_dev]
                    train_samples = train_samples[:num_train]
                    logger.info("Dev samples: %d" % len(dev_samples))
                    logger.info("Train samples: %d" % len(train_samples))
                else:
                    dev_samples = None
                    logger.info("Train samples: %d" % len(train_samples))
                    logger.info("No dev samples")

                # Training

                framework.train(
                    train_samples,
                    dev_samples if dev_samples is not None else eval_samples,
                    eval_samples,
                    client_id=client_id,
                    current_round=current_round,
                )

                if not (
                    args.eval_steps is None or args.no_eval
                ):  # fl do not test on local model.
                    metrics = framework.evaluate(
                        [], eval_samples, description="Evaluating on the Test Set"
                    )
                    _keys = list(metrics.keys())
                    for m in _keys:
                        metrics["test_" + m] = metrics[m]
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate(
                            [],
                            dev_samples,
                            description="Evaluating on the Validation Set",
                        )
                        _keys = list(dev_metrics.keys())
                        for m in _keys:
                            metrics["val_" + m] = dev_metrics[m]
            else:
                assert num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)
            if metrics is not None:
                logger.info(metrics)
                wandb.log(metrics)

            if not (args.eval_steps is None or args.no_eval):
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                wandb.log(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(
                        metrics,
                        (
                            "result/"
                            + result_file_tag(args)
                            + f"-trainset{train_set_id}.json"
                            if args.result_file is None
                            else args.result_file
                        ),
                    )
            if args.trainer != "none" and args.clean_model_at_end:
                framework.delete_checkpoints()

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if num_eval is not None:
            eval_samples = framework.task.sample_subset(
                data_split="valid", seed=0, num=num_eval
            )
        else:
            eval_samples = framework.task.valid_samples
        metrics = framework.evaluate(
            train_samples, eval_samples, one_train_set_per_eval_sample=True
        )
        logger.info(metrics)
        wandb.log(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(
                metrics,
                (
                    "result/" + result_file_tag(args) + "-onetrainpereval.json"
                    if args.result_file is None
                    else args.result_file
                ),
            )
