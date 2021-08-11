from pathlib import Path
from shutil import copyfile
from tqdm import tqdm
import random
import argparse


class rearrange_nsynth:
    def __init__(self, args):

        self.args = args
        self.origin_datapaths = [
            Path(self.args.origin_dir) / Path("nsynth-train/audio"),
            Path(self.args.origin_dir) / Path("nsynth-valid/audio"),
            Path(self.args.origin_dir) / Path("nsynth-test/audio"),
        ]

        self.target_datapath = Path(self.args.target_dir) / Path("data/wav")
        self.meta_dir = Path(self.args.meta_dir) / Path("data/meta")
        self.meta_file = Path("iden_split.txt")

    def restore_nsynth(self):
        """
        Restore Nsynth dataset in terms of voxceleb dataset for "instrument verification" training.
        """

        for datapath in self.origin_datapaths:
            paths = [p for p in datapath.iterdir()]

            print(
                "Restoring data in terms of voxceleb from "
                + str(datapath).split("/")[-2]
            )

            ## Copy files
            for src in tqdm(paths):

                instrument_id, pitch, velocity_filename = src.name.split("-")

                target_dir = self.target_datapath / Path(instrument_id)
                target_filename = Path(pitch + "-" + velocity_filename)
                dst = target_dir / target_filename

                if not dst.exists():
                    Path.mkdir(target_dir, parents=True, exist_ok=True)
                    copyfile(src, dst)

    def prepare_metadata(self):
        """
        Prepare meta data for nsynth dataseet.
        """

        target_filepaths = list(Path(self.target_datapath).glob("**/*.wav"))
        meta_dst = self.meta_dir / self.meta_file

        if not Path(meta_dst).exists():
            Path.mkdir(Path(self.meta_dir), parents=True, exist_ok=True)

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

    def generate_verification_file(self):
        """
        Generate enroll and test data for nsynth dataseet.
        """
        random.seed(10)

        nsynth_test = self.origin_datapaths[2]
        nsynth_test_data = [p.name for p in Path(nsynth_test).iterdir()]

        print("Number of nsynth test data: ", len(nsynth_test_data))
        test_insts = sorted(
            list(set([inst.split("-")[0] for inst in nsynth_test_data]))
        )

        print("Number of nsynth test instrument: ", len(test_insts))

        ## Prepare trials
        veri_dst = "veri_test.txt"
        verification_file = open(veri_dst, "w")
        trials = []

        print("Preparing verification file...")

        for test_data in tqdm(nsynth_test_data):
            for n in range(4):
                test_inst_id, test_pitch, test_velocity = test_data.split("-")
                test_filename = (
                    test_inst_id + "/" + test_pitch + "-" + test_velocity
                )

                true_data = [x for x in nsynth_test_data if test_inst_id in x]
                false_data = [
                    x for x in nsynth_test_data if test_inst_id not in x
                ]

                chosen_true = random.choice(true_data)
                true_inst_id, true_pitch, true_velocity = chosen_true.split("-")
                chosen_true = (
                    true_inst_id + "/" + true_pitch + "-" + true_velocity
                )

                chosen_false = random.choice(false_data)
                false_inst_id, false_pitch, false_velocity = chosen_false.split(
                    "-"
                )
                chosen_false = (
                    false_inst_id + "/" + false_pitch + "-" + false_velocity
                )

                trials.append("1 " + test_filename + " " + chosen_true + "\n")
                trials.append("0 " + test_filename + " " + chosen_false + "\n")

        verification_file.writelines(trials)
        verification_file.close()

    def run(self):
        self.restore_nsynth()
        self.prepare_metadata()
        self.generate_verification_file()


def main():
    """
    Usage:
    python nsynth_rearrange.py -origin_dir Nsynth/ -target_dir data/ -meta_dir data/
    """

    parser = argparse.ArgumentParser(
        description="Set configs to rearrange nsynth."
    )

    parser.add_argument(
        "-origin_dir", type=str, default="/mnt/md1/datasets/Nsynth"
    )
    parser.add_argument(
        "-target_dir",
        type=str,
        default="/mnt/md1/user_victor/speechbrain/recipes/mu_vector",
    )
    parser.add_argument(
        "-meta_dir",
        type=str,
        default="/mnt/md1/user_victor/speechbrain/recipes/mu_vector",
    )

    args = parser.parse_args()

    rearrange = rearrange_nsynth(args)
    rearrange.run()


if __name__ == "__main__":
    main()
