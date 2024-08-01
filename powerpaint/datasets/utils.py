import os


SUPPORTED_SUFFIX = [
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "png",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
]
SAVE_LOG = os.environ.get("SAVE_LOG", False)
LOG_DIR = os.environ.get("LOG_DIR", "./logs")


def save_log(log):
    if not SAVE_LOG:
        return

    worker_info = get_worker_info()
    worker_id = 0 if worker_info is None else worker_info.id
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(osp.join(LOG_DIR, f"worker_{worker_id}.log"), "a") as f:
        f.write(log)
        f.write("\n")
