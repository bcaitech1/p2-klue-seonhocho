import os

from pororo import Pororo
from tqdm import tqdm


def add_typed_entity_marker(
    s: str,
    e1: str,
    e2: str,
    e1_from: str,
    e1_to: str,
    e2_from: str,
    e2_to: str,
    ner: Pororo,
) -> str:
    ner_e1 = ner(e1)[0][1]
    ner_e2 = ner(e2)[0][1]

    e1_from, e1_to, e2_from, e2_to = map(int, [e1_from, e1_to, e2_from, e2_to])

    return (
        f"{s[:e1_from]}@*{ner_e1}*{s[e1_from:e1_to+1]}@{s[e1_to+1:e2_from]}#∧{ner_e2}∧{s[e2_from:e2_to+1]}#{s[e2_to+1:]}"
        if e1_from < e2_from
        else f"{s[:e2_from]}#∧{ner_e2}∧{s[e2_from:e2_to+1]}#{s[e2_to+1:e1_from]}@*{ner_e1}*{s[e1_from:e1_to+1]}@{s[e1_to+1:]}"
    )


if __name__ == "__main__":

    src_file = "/opt/ml/input/data/test/test.tsv"
    dst_file = "test-with-marker.tsv"

    ner = Pororo(task="ner", lang="ko")

    with open(src_file, "r") as f, open(dst_file, "w") as g:

        dumps = []
        for line in tqdm(f.readlines()):
            idx, sentence, e1, e1_from, e1_to, e2, e2_from, e2_to, label = line.split(
                "\t"
            )
            sentence = sentence.strip('"')
            marked_sentence = add_typed_entity_marker(
                sentence, e1, e2, e1_from, e1_to, e2_from, e2_to, ner
            )
            dumps.append(
                "\t".join(
                    [
                        idx,
                        marked_sentence,
                        e1,
                        e1_from,
                        e1_to,
                        e2,
                        e2_from,
                        e2_to,
                        label,
                    ]
                )
            )

        g.write("".join(dumps))

