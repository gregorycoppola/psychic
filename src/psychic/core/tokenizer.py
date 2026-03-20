"""
Minimal GPT-2 BPE tokenizer.
Reads vocab.json and merges.txt from cache.
No dependencies beyond the standard library and json.
"""
import json
import regex as re
from pathlib import Path


# GPT-2 uses this specific regex to split text into words before BPE
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""


def bytes_to_unicode():
    """
    GPT-2 maps raw bytes to unicode characters to avoid whitespace/control issues.
    Returns a dict mapping int -> str.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer:
    def __init__(self, vocab_path: Path, merges_path: Path):
        with open(vocab_path) as f:
            self.encoder = json.load(f)  # str -> int
        self.decoder = {v: k for k, v in self.encoder.items()}  # int -> str

        with open(merges_path) as f:
            lines = f.read().splitlines()
        # skip header line
        merges = [tuple(line.split()) for line in lines if line and not line.startswith("#")]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.cache = {}

    def get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token

        while True:
            # find the highest priority merge
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        token_ids = []
        for chunk in re.findall(GPT2_PATTERN, text):
            # encode chunk as bytes, map to unicode
            chunk_bytes = chunk.encode("utf-8")
            chunk_unicode = "".join(self.byte_encoder[b] for b in chunk_bytes)
            # apply BPE
            bpe_tokens = self.bpe(chunk_unicode).split(" ")
            token_ids.extend(self.encoder[t] for t in bpe_tokens)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        text = "".join(self.decoder[i] for i in token_ids)
        return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")