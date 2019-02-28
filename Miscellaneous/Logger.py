import logging
from Miscellaneous import OutputDir


class Logger(object):

    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

    hdlr = logging.FileHandler(OutputDir.defaultOutputDir + 'project.log')
    logger = logging.getLogger('hgPredictionModel')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    @staticmethod
    def time_recorder(func):

        def wrapper(*arg, **kw):
            logging.info("Begin to execute function: %s" % str(func.__qualname__))
            func(*arg, **kw)
            logging.info("Finish executing function: %s" % func.__qualname__)
        return wrapper
