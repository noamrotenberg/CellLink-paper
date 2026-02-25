import gzip
import json
import pathlib
import math
import time
import logging
import os

import numpy as np
import torch
import transformers
import tokenizers
from transformers import AutoTokenizer, AutoModel

# Configure basic logging from console
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

eps = 1e-12
models = dict()
DEFAULT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_MAX_TERM_LEN = 50
DEFAULT_BATCH_SIZE = 128
TEST_TERMS = ['MME+', 'ILCp', 'CD11blo', 'Tfh', 'Ly49D-negative', 'CD45RA+', 'CD11b+ F4/80int', 'pre-BII', 'CD14-low', 'STMN', 'IL-6 receptor α']
UNK_LIMIT = 0.0001


# --------------------------------
# Helper loaders
# --------------------------------
# NOTE: This forces lower casing
def _load_model(model_name: str):
    if model_name in models:
        tokenizer, model = models[model_name]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        model = AutoModel.from_pretrained(model_name)
        models[model_name] = (tokenizer, model)
    return tokenizer, model

class TermEncoder:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_term_length: int = DEFAULT_MAX_TERM_LEN,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        # Model / tokenizer
        tokenizer, model = _load_model(model_name)
        # — device —
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.max_term_length = max_term_length

        self.vector_map = dict()
        self._unk_statistics(TEST_TERMS)

    def dimension(self):
        if not self.vector_map:
            return None
        return next(iter(self.vector_map.values())).shape[0]

    def load(
        self,
        term_texts_path: str | pathlib.Path,
        vectors_path: str | pathlib.Path,
    ):
        term_texts_path = pathlib.Path(term_texts_path)
        vectors_path = pathlib.Path(vectors_path)
        if not term_texts_path.exists() or not vectors_path.exists():
            return
        open_func = gzip.open if term_texts_path.suffix == ".gz" else open
        with open_func(term_texts_path, "rt") as file:
            terms = json.load(file)
        vectors = np.load(vectors_path, mmap_mode="r")

        if len(terms) != vectors.shape[0]:
            raise ValueError(
                f"Mismatch: {len(terms)} terms vs {vectors.shape[0]} vectors in {vectors_path}"
            )

        # Optional: enforce a consistent dimension
        existing_dim = self.dimension()
        if existing_dim and existing_dim != vectors.shape[1]:
            raise ValueError(
                f"Loaded vector dim {vectors.shape[1]} != existing cache dim {existing_dim}"
            )

        # Store views; they keep the memmap alive
        for term, vector in zip(terms, vectors):
            self.vector_map[term] = vector

    def save(
        self,
        term_texts_path: str | pathlib.Path,
        vectors_path: str | pathlib.Path,
        *,
        sort_terms: bool = True,
    ):
        term_texts_path = pathlib.Path(term_texts_path)
        open_func = gzip.open if term_texts_path.suffix == ".gz" else open

        terms = list(self.vector_map.keys())
        if sort_terms:
            terms.sort()

        vector_list = [np.asarray(self.vector_map[t], dtype=np.float32) for t in terms]
        if not vector_list:
            # Nothing to write
            with open_func(term_texts_path, "wt") as file:
                json.dump([], file, indent=3)
            np.save(vectors_path, np.zeros((0, 0), dtype=np.float32))
            return

        vectors = np.stack(vector_list, axis=0)

        with open_func(term_texts_path, "wt") as file:
            json.dump(terms, file, indent=3)

        np.save(vectors_path, vectors)

    def encode_terms(self, terms: list[str]) -> np.ndarray:
        if not terms:
            # Mirror _encode_direct behavior
            return self._encode_direct([])

        missing = [t for t in terms if t not in self.vector_map]
        if missing:
            self._encode_direct(missing)

        # At this point, all requested terms should be in cache
        vector_list = [self.vector_map[t] for t in terms]
        # Handle the (unlikely) case of mixed dtypes/shapes early
        return np.stack([np.asarray(v, dtype=np.float32) for v in vector_list], axis=0)

    def _encode_direct(self, terms: list[str]) -> np.ndarray:
        if not terms:
            dim = self.dimension()
            if dim is None:
                dim = 0
            return np.zeros((0, dim), dtype=np.float32)

        self._log_environment()
        self._unk_statistics(terms)
        vector_list: list[np.ndarray] = []
        start_time = time.time()
        batch_count = math.ceil(len(terms) / self.batch_size)

        for batch_idx, batch_start in enumerate(
            range(0, len(terms), self.batch_size), start=1
        ):
            batch_terms = terms[batch_start : batch_start + self.batch_size]

            # Prefer the modern call style; pad to longest in batch for speed
            tokens = self.tokenizer(
                batch_terms,
                padding=True,
                truncation=True,
                max_length=self.max_term_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**tokens).last_hidden_state
                # CLS pooling (token 0); switchable later if you add pooling option
                vector_batch = outputs[:, 0, :]

            vector_batch = vector_batch.detach().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(vector_batch, axis=1, keepdims=True)
            vector_batch = vector_batch / np.clip(norms, eps, None)

            vector_list.append(vector_batch)

            if batch_count > 1:
                elapsed = time.time() - start_time
                avg = elapsed / batch_idx
                remaining = (batch_count - batch_idx) * avg
                logging.info(
                    f"Encoding batch {batch_idx}/{batch_count} on {self.device} - ETA: {remaining:.2f}s"
                )

        vectors = np.concatenate(vector_list, axis=0)
        for term, vector in zip(terms, vectors):
            self.vector_map[term] = vector
        return vectors

    def _log_environment(self):
        logging.info(f"TermEncoder env: transformers: {transformers.__version__}")
        logging.info(f"TermEncoder env: torch: {torch.__version__}")
        logging.info(f"TermEncoder env: tokenizers: {tokenizers.__version__}")
        logging.info(f"TermEncoder env: HF_HOME: {os.environ.get('HF_HOME')}", )
        logging.info(f"TermEncoder env: TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
        logging.info(f"TermEncoder env: HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")

        logging.info(f"TermEncoder env: tokenizer class: {self.tokenizer.__class__.__name__}")
        logging.info(f"TermEncoder env: model class: {self.model.__class__.__name__}")
        logging.info("TermEncoder env: model name_or_path: {}".format(getattr(self.model.config, "_name_or_path", None)))
        logging.info("TermEncoder env: model commit hash: {}".format(getattr(self.model.config, "_commit_hash", None)))

        logging.info(f"TermEncoder env: is_fast={self.tokenizer.is_fast}")
        logging.info("TermEncoder env: do_lower_case={}".format(getattr(self.tokenizer, "do_lower_case", None)))
        logging.info("TermEncoder env: init_kwargs={}".format(getattr(self.tokenizer, "init_kwargs", None)))
        logging.info(f"TermEncoder env: special_tokens_map={self.tokenizer.special_tokens_map}")

        # tokenizer fingerprints that change if tokenizer.json / vocab changes
        for name in TEST_TERMS:
            s = self.tokenizer(name, add_special_tokens=True)["input_ids"]
            logging.info("TermEncoder env: example text: {} token ids: {} len={}".format(name, s[:30], len(s)))

    def _unk_statistics(self, terms):
        total_tokens = 0
        total_unk = 0
        terms_with_unk = 0
        
        for term in terms:
            tokens = self.tokenizer.tokenize(term)
            unk_count = tokens.count(self.tokenizer.unk_token)
            
            total_tokens += len(tokens)
            total_unk += unk_count
            
            if unk_count > 0:
                terms_with_unk += 1
        
        unk_rate = total_unk / total_tokens
        print(f"UNK tokens: {total_unk}, total tokens: {total_tokens}, rate: {unk_rate:.4f}")
        print(f"Terms containing UNK: {terms_with_unk}, total terms: {len(terms)}, rate: {terms_with_unk / len(terms):.4f}")
        if unk_rate > UNK_LIMIT:
            logging.error("The tokenization is returning too many unknown tokens ([UNK]).")
            logging.error(f"The unknown token rate *should* be 0.0 but is {unk_rate}. This will cause significant performance degradation.")
            logging.error("This can be caused by a model that requires lower case being misconfigured AND receiving upper case text.")
            raise RuntimeError(f"Unknown token rate exceeds {UNK_LIMIT}: {unk_rate}")
        return unk_rate

    def add_vectors(
        self, terms: list[str], vectors: np.ndarray, *, normalize: bool = False
    ) -> None:
        if len(terms) != vectors.shape[0]:
            raise ValueError("terms and vectors length mismatch")
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, eps, None)
        for t, v in zip(terms, vectors):
            self.vector_map[t] = np.asarray(v, dtype=np.float32)

    def has(self, term: str) -> bool:
        return term in self.vector_map

if __name__ == "__main__":
    term_encoder = TermEncoder()
    term_encoder._log_environment()
    term_encoder._unk_statistics(TEST_TERMS)
    
