import logging


def get_logger(name):
    """
    Returns an already configured logger for a specific module.
    (This should be used instead of stdout.)
    :param name: the name of the modeule where the logger is created
    :return: a custom configured logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("heart.log", mode='a')
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s -- %(msg)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger.addHandler(handler)
    return logger


def progress_bar(current, total, bins):
    freq = total / bins
    bar = "#" * int(current / freq) + " " * (bins - int(current / freq))
    print("\rLoading [{}] {} %".format(bar, int(current/total * 100.0)), end="", flush=True)
    #print("\rLoading [{}] {} %".format(bar, int(current / total * 100.0)), flush=True)
    if current == total:
        print("\nLoading finsihed\n")
