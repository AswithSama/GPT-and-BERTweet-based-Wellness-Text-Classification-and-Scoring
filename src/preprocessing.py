# src/preprocessing.py

from __future__ import annotations

import re
import emoji
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import AutoTokenizer
from typing import Iterable, Optional, Set, List


# ----------------------------
# Emoji regex (keeps most Unicode emojis)
# ----------------------------
EMOJI_PATTERN = re.compile(
    "[" 
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map
    u"\U0001F1E0-\U0001F1FF"  # flags
    u"\U00002700-\U000027BF"  # dingbats
    u"\U0001F900-\U0001F9FF"  # supplemental symbols
    u"\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=re.UNICODE
)


def build_nlp(model: str = "en_core_web_sm"):
    """
    Load spaCy pipeline with heavy components disabled for speed.
    """
    return spacy.load(model, disable=["ner", "parser", "textcat"])


def build_stopwords() -> Set[str]:
    """
    Custom stopword set: keep negations + context/perspective words.
    """
    negations = {"no", "not", "nor", "n't"}

    context_words = {
        # First-person
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
        # Second-person
        "you", "your", "yours", "yourself", "yourselves",
        # Third-person singular
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        # Third-person plural
        "they", "them", "their", "theirs", "themselves",
        # Demonstratives
        "this", "that", "these", "those",
        # Interrogatives / relatives
        "who", "whom", "whose", "which", "what",
        # Possessive determiner
        "own",
        # Discourse/context markers
        "here", "there", "where", "when", "why", "how",
    }

    return set(STOP_WORDS) - negations - context_words


def emoji_to_text(e: str) -> str:
    """
    Convert emoji to its text alias using emoji.demojize
    e.g., 😂 -> ":face_with_tears_of_joy:"
    """
    return emoji.demojize(e)


def is_emoji(token: str) -> bool:
    return bool(EMOJI_PATTERN.fullmatch(token))


def normalize_punct(tok: str) -> Optional[str]:
    """
    Normalize emotional punctuation runs.
    NOTE: In your notebook you kept only '!', '?', and '...'.
    This helper is provided if you want to normalize runs like '!!!!!' -> '!'.
    """
    if set(tok) == {"!"}:
        return "!"
    if set(tok) == {"?"}:
        return "?"
    if set(tok) == {"."} and len(tok) >= 3:
        return "..."
    return None


def in_bertweet(token: str, tokenizer: AutoTokenizer) -> bool:
    """
    Check if token is in BERTweet vocab (not [UNK]).
    """
    return tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id


def preprocess_doc(
    doc,
    tokenizer: AutoTokenizer,
    stop_words: Set[str],
) -> List[str]:
    """
    spaCy Doc -> list of cleaned tokens following your exact logic.
    """
    tokens: List[str] = []
    last_punct = None   # collapse !!!!! or ??? or ...
    last_emoji = None   # collapse repeated emojis

    for tok in doc:
        t = tok.text
        lower = t.lower()

        # EMOJIS (keep one per run)
        if is_emoji(t):
            alias = emoji_to_text(t)   # 😂 -> ":face_with_tears_of_joy:"
            if alias != last_emoji and in_bertweet(alias, tokenizer):
                tokens.append(alias)
                last_emoji = alias
            last_punct = None
            continue

        # PUNCTUATION (keep ! ? ...)
        if tok.is_punct:
            if t in {"!", "?", "..."}:
                if t != last_punct:
                    tokens.append(t)
                    last_punct = t
            else:
                last_punct = None
            last_emoji = None
            continue

        # NEGATION handling: didn't -> not
        if lower in {"n't", "'nt"}:
            tokens.append("not")
            last_punct = None
            last_emoji = None
            continue

        # NUMBERS (keep)
        if tok.like_num:
            tokens.append(lower)
            last_punct = None
            last_emoji = None
            continue

        # WORDS
        if tok.is_alpha:
            lemma = tok.lemma_.lower()
            if lemma == "-pron-":  # spaCy quirk
                lemma = lower

            if lemma not in stop_words and in_bertweet(lemma, tokenizer):
                tokens.append(lemma)

            last_punct = None
            last_emoji = None
            continue

        # reset runs for anything else
        last_punct = None
        last_emoji = None

    return tokens


def preprocess_texts(
    texts: Iterable[str],
    nlp=None,
    tokenizer_name: str = "vinai/bertweet-base",
    batch_size: int = 512,
) -> List[str]:
    """
    List/Series of raw texts -> list of processed strings (" ".join(tokens)).
    """
    if nlp is None:
        nlp = build_nlp()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    stop_words = build_stopwords()

    outputs: List[str] = []
    for doc in nlp.pipe((str(t) for t in texts), batch_size=batch_size):
        toks = preprocess_doc(doc, tokenizer=tokenizer, stop_words=stop_words)
        outputs.append(" ".join(toks))
    return outputs


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "cleaned_text",
    drop_cols: Optional[List[str]] = None,
    tokenizer_name: str = "vinai/bertweet-base",
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Returns a dataframe with:
    - output_col: processed text
    - all label columns preserved
    """
    if drop_cols is None:
        drop_cols = ["text"]

    nlp = build_nlp()
    processed_texts = preprocess_texts(
        df[text_col].astype(str).tolist(),
        nlp=nlp,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
    )

    wellness_cols = df.drop(columns=drop_cols).columns
    processed_df = pd.DataFrame({output_col: processed_texts}, index=df.index)
    processed_df = pd.concat([processed_df, df[wellness_cols]], axis=1)
    return processed_df
