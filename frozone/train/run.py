import gc
import os

import torch
from pelutils import log
from pelutils.parser import Parser, Argument, Option

import frozone.train
from frozone.train.train import train


options = (
    Argument("env"),
    Argument("data-path"),
    Option("phase", default="Cone"),

    # Training parameters
    Option("dt", type=float, default=6),
    Option("history-window", type=float, default=30),
    Option("prediction-window", type=float, default=20),
    Option("batches", type=int, default=10000),
    Option("batch-size", type=int, default=200),
    Option("lr", type=float, default=1e-5),
    Option("num-models", type=int, default=1),
    Option("max-num-data-files", type=int, default=0),
    Option("eval-size", type=int, default=4000),
    Option("alpha", type=float, default=0.5),
    Option("epsilon", type=float, default=0),

    # Model parameters
    Option("dz", type=int, default=300),
    Option("encoder-name", default="FullyConnected"),
    Option("decoder-name", default="FullyConnected"),
    Option("fc-layer-num", type=int, default=3),
    Option("fc-layer-size", type=int, default=400),
    Option("dropout", type=float, default=0),
    Option("activation-fn", default="ReLU"),
)

if __name__ == "__main__":

    parser = Parser(*options, multiple_jobs=True)
    jobs = parser.parse_args()

    for job in jobs:
        job.prepare_directory()
        log.configure(os.path.join(job.location, "train.log"))
        with log.log_errors:
            log.section("Starting training of job %s" % job.name)
            try:
                frozone.train.is_doing_training = True
                train(job)
            finally:
                frozone.train.is_doing_training = False

            # Based on sad experiences in the past, the following may or may not be necessary
            # to prevent a memory leak when running multiple jobs.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
