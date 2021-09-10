from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
import argparse
import csv

validation_tracks = [
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
]


class rearrange_musdb:
    def __init__(self, args):

        self.args = args

        self.origin_datapaths = [
            Path(self.args.origin_dir) / Path("train"),
            Path(self.args.origin_dir) / Path("test"),
        ]

        self.target_datapath = Path(self.args.target_dir) / Path("data/wav")
        self.meta_dir = Path(self.args.target_dir) / Path("data/meta")
        self.iden_split_file = Path("iden_split.txt")
        self.meta_file = Path("meta.csv")

    def prepare_identity_splits(self):
        """
        Prepare identity_splits for musdb dataseet.
        """

        ## Prepare identity splits.
        target_filepaths = list(Path(self.target_datapath).glob("**/*.wav"))
        iden_split_dst = self.meta_dir / self.iden_split_file

        if not Path(iden_split_dst).exists():
            Path.mkdir(Path(self.meta_dir), parents=True, exist_ok=True)

        iden_file = open(iden_split_dst, "w")
        text = []
        index = 1

        print("Preparing id list...")
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

        iden_file.writelines(text)
        iden_file.close()  # to change file access modes

    def restore_musdb(self):
        """
        Restore MUSDB dataset in terms of voxceleb dataset for "instrument identification" training.
        """

        csv_output = [["ID", "split"]]
        entry = []

        # Copy files
        print("Copy files...")
        for datapath in self.origin_datapaths:
            paths = [
                p
                for p in datapath.glob("*/*.wav")
                if "vocals.wav" in str(p)
                or "drums.wav" in str(p)
                or "bass.wav" in str(p)
                or "other.wav" in str(p)
            ]
            print(
                "Restoring data in terms of voxceleb from "
                + str(datapath).split("/")[-2]
                + " "
                + str(datapath).split("/")[-1]
            )

            for src in tqdm(paths):

                instrument_id, song_id = (
                    str(src).split("/")[-1].replace(".wav", ""),
                    str(src).split("/")[-2],
                )

                target_dir = self.target_datapath / Path(instrument_id)
                target_filename = Path(song_id + ".wav")
                dst = target_dir / target_filename

                if not dst.exists():
                    Path.mkdir(target_dir, parents=True, exist_ok=True)
                    copyfile(src, dst)

                ## Prepare train, valid, test metadata
                meta_dst = self.meta_dir / self.meta_file

                if not Path(meta_dst).exists():
                    Path.mkdir(Path(self.meta_dir), parents=True, exist_ok=True)

                split = "train" if "train" in str(src) else "test"
                if song_id in validation_tracks:
                    split = "valid"

                csv_line = [
                    instrument_id + "/" + song_id + ".wav",
                    split,
                ]
                entry.append(csv_line)

        csv_output = csv_output + entry

        # Writing the csv lines
        with open(meta_dst, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

    def run(self):
        self.restore_musdb()
        self.prepare_identity_splits()


def main():
    """
    Usage:
    python musdb_rearrange.py -origin_dir /mnt/md1/datasets/musdb18_no_silence -target_dir musdb_no_silence
    """

    parser = argparse.ArgumentParser(
        description="Set configs to rearrange musdb."
    )

    parser.add_argument(
        "-origin_dir", type=str, default="/mnt/md1/datasets/musdb18_no_silence"
    )
    parser.add_argument(
        "-target_dir", type=str, default="musdb_no_silence",
    )

    args = parser.parse_args()

    rearrange = rearrange_musdb(args)
    rearrange.run()


if __name__ == "__main__":
    main()
