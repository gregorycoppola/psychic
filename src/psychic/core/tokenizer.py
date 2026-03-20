"""
Tokenizer support for GPT-2 BPE and Qwen2/tokenizer.json formats.
No dependencies beyond the standard library and json/regex.
"""
import json
import regex as re
from pathlib import Path


# GPT-2 BPE tokenizer
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""


def bytes_to_unicode():
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
    """GPT-2 style BPE tokenizer using vocab.json + merges.txt."""

    def __init__(self, vocab_path: Path, merges_path: Path):
        with open(vocab_path) as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path) as f:
            lines = f.read().splitlines()
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
            chunk_bytes = chunk.encode("utf-8")
            chunk_unicode = "".join(self.byte_encoder[b] for b in chunk_bytes)
            bpe_tokens = self.bpe(chunk_unicode).split(" ")
            token_ids.extend(self.encoder[t] for t in bpe_tokens)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        text = "".join(self.decoder[i] for i in token_ids)
        return bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")


class TokenizerJsonTokenizer:
    """
    Qwen2/modern tokenizer using a single tokenizer.json file.
    Handles BPE with the HuggingFace tokenizer.json format.
    """

    def __init__(self, tokenizer_path: Path):
        with open(tokenizer_path) as f:
            data = json.load(f)

        model = data["model"]
        self.encoder = model["vocab"]  # str -> int
        self.decoder = {v: k for k, v in self.encoder.items()}

        merges = model.get("merges", [])
        self.bpe_ranks = {}
        for i, merge in enumerate(merges):
            if isinstance(merge, str):
                parts = merge.split(" ", 1)
                if len(parts) == 2:
                    self.bpe_ranks[tuple(parts)] = i
            elif isinstance(merge, list) and len(merge) == 2:
                self.bpe_ranks[tuple(merge)] = i

        # added tokens
        self.added_tokens = {}
        for entry in data.get("added_tokens", []):
            self.added_tokens[entry["content"]] = entry["id"]
            self.decoder[entry["id"]] = entry["content"]

        # pre-tokenizer pattern — Qwen2 uses a byte-level BPE
        # similar to GPT-2 but with different split pattern
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}

        # Qwen2 uses the same GPT2-style pattern
        self.pattern = GPT2_PATTERN

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
        for chunk in re.findall(self.pattern, text):
            chunk_bytes = chunk.encode("utf-8")
            chunk_unicode = "".join(self.byte_encoder[b] for b in chunk_bytes)
            bpe_tokens = self.bpe(chunk_unicode).split(" ")
            for t in bpe_tokens:
                if t in self.encoder:
                    token_ids.append(self.encoder[t])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        parts = []
        for i in token_ids:
            if i in self.decoder:
                parts.append(self.decoder[i])
        text = "".join(parts)
        try:
            return bytearray([self.byte_decoder.get(c, 0) for c in text]).decode("utf-8", errors="replace")
        except Exception:
            return text


def load_tokenizer(cfg: dict, cache: Path):
    """Load the right tokenizer for a model config."""
    tokenizer_type = cfg.get("tokenizer_type", "bpe")
    if tokenizer_type == "tokenizer_json":
        path = cache / cfg["tokenizer_filename"]
        return TokenizerJsonTokenizer(path)
    else:
        vocab_path = cache / cfg["vocab_filename"]
        merges_path = cache / cfg["merges_filename"]
        return BPETokenizer(vocab_path, merges_path)