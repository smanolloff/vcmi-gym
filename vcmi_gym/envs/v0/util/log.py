import logging


class RelativeTimeFormatter(logging.Formatter):
    def format(self, record):
        record.reltime = "%.2f" % (record.relativeCreated / 1000)
        return super().format(record)


def get_logger(name, level):
    logger = logging.getLogger(name.split(".")[-1])
    logger.setLevel(getattr(logging, level))

    fmt = "-- %(reltime)ss (%(process)d) [%(name)s] %(levelname)s: %(message)s"
    formatter = RelativeTimeFormatter(fmt)

    loghandler = logging.StreamHandler()
    loghandler.setLevel(logging.DEBUG)
    loghandler.setFormatter(formatter)
    logger.addHandler(loghandler)

    return logger
