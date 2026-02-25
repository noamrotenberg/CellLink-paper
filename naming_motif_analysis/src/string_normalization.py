import re
import string

import PerformanceProfiler


class SpacyStringNormalizer:

    def __init__(self, nlp, maxiter=3):
        self.nlp = nlp
        self.caps = re.compile("[A-Z]*")
        self.vowels = "AEIOUaeiou"
        self.maxiter = maxiter
        self.cache = dict()

    def is_exception(self, token):
        text = token.text
        # logging.debug(f"text = {text}")
        if text.endswith("s"):
            text = text[:-1]
        if len(text) < 3:
            # logging.debug(f"--> True @1")
            return True
        # If no vowels, must be an abbreviation
        if not any(char in self.vowels for char in text):
            # logging.debug(f"--> True @2")
            return True
        # Check for initial lower, uppper pattern; like "pH"
        if text[0].islower() and text[1].isupper():
            # logging.debug(f"--> True @3")
            return True
        # Check for mixed capitalization
        # Skip the first letter - that's ok
        upper = 1 if any(char.isupper() for char in text[1:] if char.isalpha()) else 0
        lower = 1 if any(char.islower() for char in text if char.isalpha()) else 0
        digit = 1 if any(char.isdigit() for char in text) else 0
        punct = 1 if any(char in string.punctuation for char in text) else 0
        if upper + lower + digit + punct > 1:
            # logging.debug(f"--> True @4")
            return True
        # logging.debug(f"--> False @5")
        return False

    def _internal_update(self, text):
        PerformanceProfiler.start("SpacyStringNormalizer._internal_update()")
        PerformanceProfiler.start("SpacyStringNormalizer._internal_update()@doc")
        doc = self.nlp(text)
        PerformanceProfiler.end("SpacyStringNormalizer._internal_update()@doc")
        updated = ""
        for i, token in enumerate(doc):
            if self.is_exception(token):
                # logging.debug(f"{i}(token): {token.text} --> {token.text}")
                updated += token.text
            else:
                # logging.debug(f"{i}(lemma): {token.text} --> {token.lemma_}")
                updated += token.lemma_.lower()
            if len(token.whitespace_) > 0:
                updated += " "
        updated = updated.strip()
        PerformanceProfiler.end("SpacyStringNormalizer._internal_update()")
        return updated

    def _update_iter(self, text: str) -> str:
        PerformanceProfiler.start("SpacyStringNormalizer._update_iter()")
        s = text
        seen = set()
        for i in range(self.maxiter):
            new = self._internal_update(s)
            if new == s:
                PerformanceProfiler.end("SpacyStringNormalizer._update_iter()")
                return new
            if new in seen:
                PerformanceProfiler.end("SpacyStringNormalizer._update_iter()")
                return s
            seen.add(s)
            s = new
        PerformanceProfiler.end("SpacyStringNormalizer._update_iter()")
        return s

    def update(self, text):
        PerformanceProfiler.start("SpacyStringNormalizer.update()")
        text = text.strip()
        cached = self.cache.get(text)
        if not cached is None:
            PerformanceProfiler.end("SpacyStringNormalizer.update()")
            return cached
        text = self._update_iter(text)
        self.cache[text] = text
        PerformanceProfiler.end("SpacyStringNormalizer.update()")
        return text
