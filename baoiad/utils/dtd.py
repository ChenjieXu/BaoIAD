"""Shared DTD texture dataset download utility."""

import logging
import os
import tarfile

logger = logging.getLogger(__name__)

DTD_DOWNLOAD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_DEFAULT_DIR = "data/dtd"


def download_dtd(target_dir: str = DTD_DEFAULT_DIR) -> str:
    """Download and extract DTD texture dataset if not present.

    Args:
        target_dir: Directory to store the dataset.

    Returns:
        Path to the DTD images directory.
    """
    dtd_images_dir = os.path.join(target_dir, 'dtd', 'images')
    if os.path.isdir(dtd_images_dir):
        return dtd_images_dir

    os.makedirs(target_dir, exist_ok=True)
    tar_path = os.path.join(target_dir, "dtd-r1.0.1.tar.gz")

    if not os.path.isfile(tar_path):
        import socket
        import urllib.request
        logger.info("Downloading DTD texture dataset...")
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(60)
        try:
            urllib.request.urlretrieve(DTD_DOWNLOAD_URL, tar_path)
        except Exception as e:
            if os.path.isfile(tar_path):
                os.remove(tar_path)
            raise RuntimeError(
                f"Failed to download DTD dataset: {e}. "
                f"Please manually download from {DTD_DOWNLOAD_URL}"
            )
        finally:
            socket.setdefaulttimeout(old_timeout)

    logger.info("Extracting DTD texture dataset...")
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(target_dir)

    if os.path.isfile(tar_path):
        os.remove(tar_path)

    if not os.path.isdir(dtd_images_dir):
        raise FileNotFoundError(
            f"DTD dataset extraction failed. Expected images at {dtd_images_dir}. "
            f"Please manually download from {DTD_DOWNLOAD_URL} and extract to {target_dir}/"
        )

    logger.info(f"DTD dataset ready at {dtd_images_dir}")
    return dtd_images_dir
