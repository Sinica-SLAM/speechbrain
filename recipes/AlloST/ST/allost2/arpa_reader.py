from typing import Dict, List, Tuple


def ngam_information_to_int(ngram_information: str):
    """Extract ngram information"""
    number_of_ngram = ngram_information.strip().split("=")[-1]
    return int(number_of_ngram)


def read_arpa(arpa_file_path: str) -> Dict[str, int]:
    """Read the given arpa LM, and return a word-score mapping"""
    with open(arpa_file_path, "r", encoding="utf-8") as arpa_file:
        arpa_lines = arpa_file.readlines()

        ngram_numbers = list(
            filter(lambda line: line.startswith("ngram"), arpa_lines)
        )
        ngram_numbers = list(map(ngam_information_to_int, ngram_numbers))

        ngram_scores = {}

        # space + \data\ ngrams' info + space + \1-grams
        ngram_start_index = 1 + 1 + len(ngram_numbers) + 1 + 1
        for ngram_number in ngram_numbers:
            ngrams = arpa_lines[
                ngram_start_index : ngram_start_index + ngram_number
            ]
            ngram_start_index += ngram_number + 1 + 1

            for ngram in ngrams:
                ngram = ngram.strip().split("\t")
                ngram_scores[ngram[1]] = float(ngram[0])

    return ngram_scores


def find_the_possible_path(
    paths: List[List[Tuple[any]]],
    sequence_length: int,
    symbol: str,
    score: float,
    current_start: int,
    current_end: int,
) -> List[List[Tuple[any]]]:
    """Find all possible paths by the given n-gram token"""
    is_found_position_to_insert = False
    for path_index, path in enumerate(paths):
        # if this path already meets the end
        last_position_pluse_current_token_length = paths[path_index][-1][3] + (
            current_end - current_start
        )
        if last_position_pluse_current_token_length > (sequence_length - 1):
            continue

        is_overlapping = False
        # check if overlapping
        for vocab in path:
            start = vocab[2]
            end = vocab[3]
            vocab_span_index = [i for i in range(start, end + 1)]
            current_vocab_span_index = [
                i for i in range(current_start, current_end)
            ]
            overlapping = set(vocab_span_index) & set(current_vocab_span_index)

            if len(overlapping):
                is_overlapping = True
                break

        if not is_overlapping:
            paths[path_index].append(
                (symbol, score, current_start, current_end - 1)
            )
            is_found_position_to_insert = True
            break

    if not is_found_position_to_insert:
        paths.append([(symbol, score, current_start, current_end - 1)])

    return paths


def find_all_possible_vocabs(
    sequence: str,
    ngram_scores: Dict[str, int],
    order: int = 3,
    threshold: float = -2,
) -> List[List[str]]:
    """Find all possibl vocab from the given sequence
    and put them togather a possible phone sequence

    Example:
    'ɒ ʝ a iː o' can be tokenized into ['ʝ a', 'iː o', 'ɒ ʝ a', 'a iː o']
    and the possible sequences could be
    ['ɒ', 'ʝ a', 'iː o'],
    ['ɒ ʝ a', 'iː', 'o'],
    ['ɒ', 'ʝ', 'a iː o'],
    """
    phone_symbols = sequence.split(" ")

    paths = []
    sequence_length = len(sequence.split(" "))

    # Find all possible vocabs
    for ngram in range(2, order + 1):
        last_index = -ngram + 1
        for index, phone_symbol in enumerate(phone_symbols[:last_index]):
            symbol = (
                phone_symbol
                + " "
                + " ".join(phone_symbols[index + 1 : index + ngram])
            )

            score = ngram_scores.get(symbol, -999)
            if score > threshold:
                if len(paths) == 0:
                    paths.append([(symbol, score, index, index + ngram - 1)])
                else:
                    paths = find_the_possible_path(
                        paths=paths,
                        sequence_length=sequence_length,
                        symbol=symbol,
                        score=score,
                        current_start=index,
                        current_end=index + ngram,
                    )
    # Put possible vocabs togather
    results = [[] for _ in range(len(paths))]
    sequence_index = [i for i in range(sequence_length)]

    for index, path in enumerate(paths):
        for symbol in path:
            results[index].append(symbol[0])

        phone_index = list(
            map(lambda p: [i for i in range(p[2], p[3] + 1)], path)
        )
        phone_index = [index for vocab in phone_index for index in vocab]

        pad_indices = set(sequence_index) - set(phone_index)
        pad_indices = list(pad_indices)

        for pad_index in pad_indices:
            results[index].insert(pad_index, phone_symbols[pad_index])

    possible_path_number = sum([i for i in range(2, order + 1)])

    # Pad the not enough samples
    if len(results) < possible_path_number:
        not_enough_number = possible_path_number - len(results)
        for _ in range(not_enough_number):
            results.append(phone_symbols.copy())

    return results


def ngram2group_index(sequence: List[str]) -> List[int]:
    """Make the vocab with the same index in original sequence

    Example:
    ['ɒ', 'ʝ a', 'iː o'] -> [0, 1, 1, 2, 2]

    'ʝ a' is a valid vocab, so its range (1 and 2 in this example) share the same group id
    'iː o' also share the same group id 2 in range (3 and 4)
    """
    ngrams = list(map(lambda token: len(token.split(" ")), sequence))

    group_indexs = []
    group_index = 1

    for ngram in ngrams:
        group_indexs += [group_index] * ngram
        group_index += 1

    return group_indexs


def sequence2group_index(
    sequence: str, ngram_scores: str, order: int = 3, threshold: float = -2,
) -> List[List[int]]:
    """Map the given sequence to group id"""
    vocabs = find_all_possible_vocabs(
        sequence=sequence,
        ngram_scores=ngram_scores,
        order=order,
        threshold=threshold,
    )

    group = []
    for vocab in vocabs:
        index = ngram2group_index(vocab)
        group.append(index)

    return group
