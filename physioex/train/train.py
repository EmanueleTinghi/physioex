import argparse
import importlib

import yaml

from physioex.train import Trainer
from physioex.train.networks import config

from loguru import logger


@logger.catch
def register_experiment(experiment: str = None):

    global config

    logger.info(f"Registering experiment {experiment}")

    try:
        with open(experiment, "r") as f:
            experiment = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Experiment {experiment} not found")

    experiment_name = experiment["name"]

    config[experiment_name] = dict()

    module = importlib.import_module(experiment["module"])

    config[experiment_name]["module"] = getattr(module, experiment["class"])
    config[experiment_name]["module_config"] = experiment["module_config"]
    config[experiment_name]["input_transform"] = experiment["input_transform"]

    if experiment["target_transform"] is not None:
        if experiment["module"] != experiment["target_transform"]["module"]:
            module = importlib.import_module(experiment["target_transform"]["module"])

        config[experiment_name]["target_transform"] = getattr(
            module, experiment["target_transform"]["function"]
        )
    else:
        logger.warning(f"Target transform not found for {experiment_name}")
        config[experiment_name]["target_transform"] = None

    return experiment_name


def main():
    parser = argparse.ArgumentParser(description="Training script")

    # experiment arguments
    parser.add_argument(
        "-e",
        "--experiment",
        default="chambon2018",
        type=str,
        help='Specify the experiment to run. Expected type: str. Default: "chambon2018"',
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        default=None,
        type=str,
        help="Specify where to save the checkpoint. Expected type: str. Default: None",
    )
    parser.add_argument(
        "-l",
        "--loss",
        default="cel",
        type=str,
        help='Specify the loss function to use. Expected type: str. Default: "cel" (Cross Entropy Loss)',
    )

    # dataset args
    parser.add_argument(
        "-d",
        "--dataset",
        default="sleep_physionet",
        type=str,
        help='Specify the dataset to use. Expected type: str. Default: "SleepPhysionet"',
    )
    parser.add_argument(
        "-v",
        "--version",
        default="2018",
        type=str,
        help='Specify the version of the dataset. Expected type: str. Default: "2018"',
    )
    parser.add_argument(
        "-p",
        "--picks",
        default="Fpz-Cz",
        type=str,
        help="Specify the signal electrodes to pick to train the model. Expected type: list. Default: 'Fpz-Cz'",
    )

    # sequence
    parser.add_argument(
        "-sl",
        "--sequence_lenght",
        default=3,
        type=int,
        help="Specify the sequence length for the model. Expected type: int. Default: 3",
    )

    # trainer
    parser.add_argument(
        "-me",
        "--max_epoch",
        default=20,
        type=int,
        help="Specify the maximum number of epochs for training. Expected type: int. Default: 20",
    )
    parser.add_argument(
        "-vci",
        "--val_check_interval",
        default=300,
        type=int,
        help="Specify the validation check interval during training. Expected type: int. Default: 300",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=32,
        type=int,
        help="Specify the batch size for training. Expected type: int. Default: 32",
    )

    parser.add_argument(
        "-nj",
        "--n_jobs",
        default=10,
        type=int,
        help="Specify the number of jobs for parallelization. Expected type: int. Default: 10",
    )

    parser.add_argument(
        "-imb",
        "--imbalance",
        default=False,
        type=bool,
        help="Specify rather or not to use f1 score instead of accuracy to save the checkpoints. Expected type: bool. Default: False",
    )

    args = parser.parse_args()

    # check if the experiment is a yaml file
    if args.experiment.endswith(".yaml") or args.experiment.endswith(".yml"):
        args.experiment = register_experiment(args.experiment)

    Trainer(
        model_name=args.experiment,
        dataset_name=args.dataset,
        ckp_path=args.checkpoint,
        loss_name=args.loss,
        version=args.version,
        picks=args.picks,
        sequence_length=args.sequence_lenght,
        max_epoch=args.max_epoch,
        val_check_interval=args.val_check_interval,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        imbalance=args.imbalance,
    ).run()


if __name__ == "__main__":
    main()