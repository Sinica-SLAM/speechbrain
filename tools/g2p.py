"""A convenience script to transcribe text into phonemes using a
pretrained Grapheme-to-Phoneme (G2P) model

The scripts to train G2P models are located in
recipes/LibriSpeech/G2P

Usage
-----
Command-line Grapheme-to-Phoneme conversion tool

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The path to the pretrained model
  --hparams HPARAMS     The name of the hyperparameter file
  --text TEXT           The text to transcribe
  --text-file TEXT_FILE
                        the text file to transcribe
  --output-file OUTPUT_FILE
                        the file to which results will be output
  -i, --interactive     Launches an interactive shell

Examples
--------
Start an interactive shell:
```bash
python g2p.py --model /path/to/model --interactive
```

Once started, the tool will display a prompt allowing you to enter
lines of text and transcribe them in real time. This is useful for
exploratory analysis (e.g. evaluating how a model resolves ambiguities)

Transcribe a single example:
```bash
python g2p.py --model /path/to/model --text "This is a line of text"
```

The tool will transcribe the single sample and output the transcribed
text to standard outout.

Transcribe a file:
python g2p.py --model /path/to/model --text-file text.txt \
    --output-file phonemes.txt

This is useful in a scenario when an entire dataset needs to be
transcribed. For instance, one may want to train a Text-to-Speech
model with phonemes, rather than raw text, as inputs. The tool
can accept text files of arbitrary size with samples recorded one
per line. Each line is a sample, and the maximum size of a line
is determined by the underlying model, as well as available
resources (RAM or GPU memory).


Authors
* Artem Ploujnikov 2021
"""
import itertools
import math
import os
import sys
import speechbrain as sb
import traceback
from cmd import Cmd
from argparse import ArgumentParser
from speechbrain.pretrained.interfaces import GraphemeToPhoneme
from tqdm.auto import tqdm

MSG_MODEL_NOT_FOUND = "Model path not found"
MSG_HPARAMS_NOT_FILE = "Hyperparameters file not found"


def transcribe_text(g2p, text):
    """
    Transcribes a single line of text and outputs it

    Arguments
    ---------
    g2p: speechbrain.pretrained.interfaces.GrpahemeToPhoneme
        a pretrained G2P model instance

    text: str
        the text to transcribe
    """
    output = g2p(text)
    print(" ".join(output))


def transcribe_file(g2p, text_file_name, output_file_name=None, batch_size=64):
    """
    Transcribes a file with one example per line

    g2p: speechbrain.pretrained.interfaces.GrpahemeToPhoneme
        a pretrained G2P model instance

    text_file_name: str
        the name of the source text file

    output_file_name: str
        the name of the output file. If omitted, the phonemes will
        be output to stdout

    batch_size: str
        the number of examples per batch


    """
    line_count = get_line_count(text_file_name)
    with open(text_file_name) as text_file:
        if output_file_name is None:
            transcribe_stream(
                g2p, text_file, sys.stdout, batch_size, total=line_count
            )
        else:
            with open(output_file_name, "w") as output_file:
                transcribe_stream(
                    g2p, text_file, output_file, batch_size, total=line_count
                )


def get_line_count(text_file_name):
    """
    Counts the lines in a file (without loading the entire file into memory)

    Arguments
    ---------
    file_name: str
        the file name

    Returns
    -------
    line_count: int
        the number of lines in the file
    """
    with open(text_file_name) as text_file:
        return sum(1 for _ in text_file)


_substitutions = {" ": "<spc>"}


def transcribe_stream(g2p, text_file, output_file, batch_size=64, total=None):
    """
    Transcribes a file stream

    Arguments
    ---------
    g2p: speechbrain.pretrained.interfaces.GrpahemeToPhoneme
        a pretrained G2P model instance
    text_file: file
        a file object from which text samples will be read
    output_file: file
        the file object to which phonemes will be output
    batch_size: 64
        the size of the batch passed to the model
    total: int
        the total number of examples (used for the progress bar)
    """
    batch_count = math.ceil(total // batch_size)
    for batch in tqdm(chunked(text_file, batch_size), total=batch_count):
        phoneme_results = g2p(batch)
        for result in phoneme_results:
            line = " ".join(
                _substitutions.get(phoneme, phoneme) for phoneme in result
            )
            print(line, file=output_file)
            output_file.flush()


def chunked(iterable, batch_size):
    """Break *iterable* into lists of length *n*:

        >>> list(chunked([1, 2, 3, 4, 5, 6], 3))
        [[1, 2, 3], [4, 5, 6]]

    By the default, the last yielded list will have fewer than *n* elements
    if the length of *iterable* is not divisible by *n*:

        >>> list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3))
        [[1, 2, 3], [4, 5, 6], [7, 8]]


    Adopted and simplified from more-itertools
    https://more-itertools.readthedocs.io/en/stable/_modules/more_itertools/more.html#chunked

    Arguments
    ---------
    iterable: iterable
        any iterable of individual samples

    batch_size: int
        the size of each chunk

    Returns
    -------
    batched_iterable: iterable
        an itearble of batches

    """
    iterable = iter(iterable)
    iterator = iter(lambda: list(itertools.islice(iterable, batch_size)), [])
    return iterator


class InteractiveG2P(Cmd):
    """An interactive G2P evaluator (useful for manually evaluating G2P sequences)

    Arguments
    ---------
    model: speechbrain.pretrained.interfaces.GrpahemeToPhoneme
        a pretrained G2P model instance
    """

    prompt = "g2p> "
    intro = """Welcome to the interactive G2P shell. Type ? to list commands.
Type text to transcribe. Type exit to quit the shell"""

    HELP_G2P = """Transcribes a text sample
Example: g2p A quick brown fox jumped over the lazy dog"""
    HELP_EXIT = "Exits the interactive G2P shell"
    MSG_ERROR = "--- G2P CONVERSION FAILED ---"
    QUIT_COMMANDS = ["", "q", "quit", "exit", "quit()"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def do_g2p(self, inp):
        """Performs G2P transcription

        Arguments
        ---------
        inp: str
            the user input
        """
        try:
            transcribe_text(self.model, inp)
        except Exception:
            print(self.MSG_ERROR)
            print(traceback.format_exc())
            print(self.MSG_ERROR)

    def do_exit(self, inp):
        """Exits the interactive shell"""
        return True

    def help_g2p(self):
        """The help text for the g2p command"""
        print(self.HELP_G2P)

    def help_exit(self):
        """The help text for the exit command"""
        print(self.HELP_EXIT)

    def default(self, inp):
        """The default input handler - exits on an empty
        input, transcribes otherwise

        Arguments
        ---------
        inp: str
            the user input
        """
        if inp.strip() in self.QUIT_COMMANDS:
            return True
        self.do_g2p(inp)


def main():
    """Runs the command-line tool"""
    # Parse command-line arguments
    parser = ArgumentParser(
        description="Command-line Grapheme-to-Phoneme conversion tool"
    )
    parser.add_argument(
        "--model", required=True, help="The path to the pretrained model"
    )
    parser.add_argument(
        "--hparams",
        help="The name of the hyperparameter file",
        default="hyperparams.yaml",
    )
    parser.add_argument("--text", help="The text to transcribe")
    parser.add_argument("--text-file", help="the text file to transcribe")
    parser.add_argument(
        "--output-file", help="the file to which results will be output"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        help="Launches an interactive shell",
        default=False,
        action="store_true",
    )

    arguments, override_arguments = parser.parse_known_args()
    _, run_opts, overrides = sb.parse_arguments(
        [arguments.hparams] + override_arguments
    )

    # Ensure the model directory exists
    if not os.path.isdir(arguments.model):
        print(MSG_MODEL_NOT_FOUND, file=sys.stderr)
        sys.exit(1)

    # Determine the path to the hyperparameters file
    hparams_file_name = os.path.join(arguments.model, arguments.hparams)
    if not os.path.isfile(hparams_file_name):
        print(MSG_HPARAMS_NOT_FILE, file=sys.stderr)
        sys.exit(1)

    # Initialize the pretrained grapheme-to-phoneme model
    g2p = GraphemeToPhoneme.from_hparams(
        hparams_file=hparams_file_name,
        source=arguments.model,
        overrides=overrides,
        run_opts=run_opts,
        savedir=arguments.model,
    )

    # Language model adjustments
    if getattr(g2p.hparams, "use_language_model", False):
        g2p.hparams.beam_searcher = g2p.hparams.beam_searcher_lm

    # Launch an interactive model
    if arguments.interactive:
        shell = InteractiveG2P(model=g2p)
        shell.cmdloop()
    # Transcribe a single line of text
    elif arguments.text:
        transcribe_text(g2p, arguments.text)
    # Transcribe a file
    elif arguments.text_file:
        transcribe_file(
            g2p=g2p,
            text_file_name=arguments.text_file,
            output_file_name=arguments.output_file,
            batch_size=g2p.hparams.eval_batch_size,
        )
    else:
        print(
            "ERROR: Either --text or --text-file is required "
            "in non-interactive mode",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()