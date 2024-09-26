import os
import re


def get_most_recent_version(base_name, directory='.') -> str:
    """Function taken from GPT to get the most recent version of a directory/file. 
    This is helpful if you train mutilple times to get the most recent pathing for 
    the training files without having to manually update the calls

    Args:
        base_name (str): base name for file
        directory (str, optional): directory to search

    Returns:
        str: path to most recent version of file
    """

    dirs = [d for d in os.listdir(directory) if os.path.isdir(
        os.path.join(directory, d)) and d.startswith(base_name)]

    if not dirs:
        return None

    most_recent_dir = max(
        dirs, key=lambda d: os.path.getmtime(os.path.join(directory, d)))

    return os.path.join(directory, most_recent_dir)
