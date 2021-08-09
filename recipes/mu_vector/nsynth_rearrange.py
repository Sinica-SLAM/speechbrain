from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

origin_datapaths = [
    "/mnt/md1/datasets/Nsynth/nsynth-train/audio",
    "/mnt/md1/datasets/Nsynth/nsynth-valid/audio",
    "/mnt/md1/datasets/Nsynth/nsynth-test/audio",
]

target_datapath = "/mnt/md1/user_victor/speechbrain/recipes/mu_vector/data/wav"


def restore_nsynth():
    """
    Restore Nsynth dataset in terms of voxceleb dataset for "instrument verification" training.
    """

    for datapath in origin_datapaths:
        paths = [p for p in Path(datapath).iterdir()]

        print(
            "Restoring data in terms of voxceleb from "
            + datapath.split("/")[-2]
        )

        ## Copy files
        for src in tqdm(paths):

            instrument_id, pitch, velocity_filename = src.name.split("-")

            target_dir = Path(target_datapath) / Path(instrument_id)
            target_filename = Path(pitch + "-" + velocity_filename)
            dst = target_dir / target_filename

            if not dst.exists():
                Path.mkdir(target_dir, parents=True, exist_ok=True)
                copyfile(src, dst)


def prepare_metadata():
    """
    Prepare meta data for nsynth dataseet.
    """

    target_filepaths = list(Path(target_datapath).glob("**/*.wav"))

    meta_dir = "/mnt/md1/user_victor/speechbrain/recipes/mu_vector/data/meta/"
    meta_file = "iden_split.txt"
    meta_dst = meta_dir + meta_file

    if not Path(meta_dst).exists():
        Path.mkdir(Path(meta_dir), parents=True, exist_ok=True)

    metafile = open(meta_dst, "w")
    text = []
    index = 1

    print("Preparing metadata...")
    for n in tqdm(range(len(target_filepaths))):
        if n > 1:
            if (
                str(target_filepaths[n]).split("/")[-2]
                != str(target_filepaths[n - 1]).split("/")[-2]
            ):
                index = 1

        ins_id, filename = str(target_filepaths[n]).split("/")[-2:]
        text.append(str(index) + " " + ins_id + "/" + filename + "\n")
        index += 1

    metafile.writelines(text)
    metafile.close()  # to change file access modes


def run():
    restore_nsynth()
    prepare_metadata()


if __name__ == "__main__":
    run()
