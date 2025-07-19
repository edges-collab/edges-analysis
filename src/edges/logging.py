"""Nicer logger."""

import logging

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
logging.STRUCTURE = 35  # "STRUCTURE" log level
logging.SUCCESS = 39
logging.addLevelName(logging.STRUCTURE, "STRUCTURE")
logging.addLevelName(logging.SUCCESS, "SUCCESS")

# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    "WARNING": YELLOW,
    "INFO": WHITE,
    "DEBUG": BLUE,
    "CRITICAL": YELLOW,
    "ERROR": RED,
    "SUCCESS": CYAN,
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        super().__init__(fmt="%(levelname)-18s| %(message)s", datefmt=None, style="%")
        self.use_color = use_color

    def format(self, record):
        orig = self._style._fmt
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = (
                COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            )
            record.levelname = levelname_color

        if record.levelname == "STRUCTURE" and self.use_color:
            self._style._fmt = BOLD_SEQ + "%(message)s" + RESET_SEQ

        result = logging.Formatter.format(self, record)
        self._style._fmt = orig
        return result


class ColoredLogger(logging.Logger):
    FORMAT = "$BOLD$RESET%(levelname)-18s| %(message)s"
    COLOR_FORMAT = formatter_message(FORMAT, True)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.WARNING)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)
        self.errored = 0

        self.addHandler(console)
        return

    def _structure(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.STRUCTURE):
            self._log(logging.STRUCTURE, msg, args, **kwargs)

    def success(self, msg, *args, **kwargs):
        self.errored -= 1
        if self.isEnabledFor(logging.SUCCESS):
            self._log(logging.SUCCESS, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.errored += 1
        if self.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, msg, args, **kwargs)


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger("edges-io")
logger.setLevel(logging.WARNING)
logging.setLoggerClass(logging.Logger)
