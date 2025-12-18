import re


token_regex = "[^\W_]+|[^\w\s]|_|,"
p_tokens = re.compile(token_regex, re.UNICODE)

def tokenize(text):
    return p_tokens.findall(text)
    
def s_stem(str):
    # NOTE: will only stem lower case, intended to leave abbreviations alone
    if not str.endswith("s"):
        return str
    if (str.endswith("viruses")):
        return re.sub("uses$", "us", str)
    # If word ends in "ies" but not "eies" or "aies" then "ies" --> "y"
    if (str.endswith("ies") and not str.endswith("eies") and not str.endswith("aies")):
        return re.sub("ies$", "y", str)
    # If a word ends in "es" but not "aes" "ees" or "oes" --> "es" --> "e"
    if (str.endswith("es") and not str.endswith("aes") and not str.endswith("ees") and not str.endswith("oes")):
        if str.endswith("sses"):
            return re.sub("es$", "", str)
        else:
            return re.sub("es$", "e", str)            
    # If a word ends in "s" but not "us" or "ss" then "s" --> null
    if (not str.endswith("us") and not str.endswith("ss")):
        return re.sub("s$", "", str)
    # Return as-is
    return str

def s_stem_all(name):
    # Note assumption that whitespace has been normalized
    words = list(map(s_stem, tokenize(name)))
    return " ".join(words)