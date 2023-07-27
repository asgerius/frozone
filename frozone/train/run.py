import os

from pelutils import log
from pelutils.parser import Parser

from frozone.train.train import train


options = ()

if __name__ == "__main__":

    parser = Parser(*options, multiple_jobs=True)
    jobs = parser.parse_args()

    for job in jobs:
        job.prepare_directory()
        log.configure(os.path.join(job.location, "train.log"))
        with log.log_errors:
            log.section("Starting training of job %s" % job.name)
            train(job)
