import os

from pelutils import log
from pelutils.parser import Parser, Argument, Option

from frozone.train.train import train


options = (
    Argument("env"),
    Argument("data-path"),

    # Training parameters
    Option("dt", type=float, default=6),
    Option("history-window", type=float, default=30),
    Option("prediction-window", type=float, default=20),
    Option("batches", type=int, default=10000),
    Option("batch-size", type=int, default=200),
    Option("lr", type=float, default=1e-5),
    Option("max-num-data-files", type=int, default=0),
    Option("eval-size", type=int, default=4000),
    Option("alpha", type=float, default=0.5),
    Option("epsilon", type=float, default=0),

    # Model parameters
    Option("Dz", type=int, default=300),
    Option("history-encoder", default="FullyConnected"),
    Option("target-encoder", default="FullyConnected"),
    Option("dynamics-network", default="FullyConnected"),
    Option("control-network", default="FullyConnected"),
    Option("fc-layer-num", type=int, default=3),
    Option("fc-layer-size", type=int, default=400),
    Option("resnext-cardinality", type=int, default=1),
    Option("dropout-p", type=float, default=0),
)

if __name__ == "__main__":

    parser = Parser(*options, multiple_jobs=True)
    jobs = parser.parse_args()

    for job in jobs:
        job.prepare_directory()
        log.configure(os.path.join(job.location, "train.log"))
        with log.log_errors:
            log.section("Starting training of job %s" % job.name)
            train(job)
