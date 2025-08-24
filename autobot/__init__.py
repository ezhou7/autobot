import os
from autobot.utils.fs import read_file


CODEBASE_PATH = os.path.dirname(os.path.abspath(__file__))
RESOURCES_PATH = os.path.join(CODEBASE_PATH, "resources")


def get_resource(filename: str):
    return read_file(os.path.join(RESOURCES_PATH, filename))
