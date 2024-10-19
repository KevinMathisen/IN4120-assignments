# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
from .ranker import Ranker
from .corpus import Corpus
from .posting import Posting
from .invertedindex import InvertedIndex


class BetterRanker(Ranker):
    """
    A ranker that does traditional TF-IDF ranking, possibly combining it with
    a static document score (if present).

    The static document score is assumed accessible in a document field named
    "static_quality_score". If the field is missing or doesn't have a value, a
    default value of 0.0 is assumed for the static document score.

    See Section 7.1.4 in https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf.
    """

    # These values could be made configurable. Hardcode them for now.
    _dynamic_score_weight = 1.0
    _static_score_weight = 1.0
    _static_score_field_name = "static_quality_score"
    _static_score_default_value = 0.0

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self._score = 0.0
        self._document_id = None
        self._corpus = corpus
        self._inverted_index = inverted_index

    def reset(self, document_id: int) -> None:
        self._document_id = document_id
        self._score = 0.0

    def update(self, term: str, multiplicity: int, posting: Posting) -> None:
        assert self._document_id == posting.document_id

        # Get numbers needed to calculate td_idf
        term_freq = posting.term_frequency
        total_num_documents = self._corpus.size()
        document_freq = self._inverted_index.get_document_frequency(term)

        tf_idf_score = math.log10(1+term_freq) * math.log10(total_num_documents/document_freq)

        # Find static score if specified, default 0
        static_doc_score = self._corpus.get_document(posting.document_id).get_field("static_quality_score", 0.0)

        self._score += (static_doc_score + tf_idf_score) * multiplicity

    def evaluate(self) -> float:
        return self._score
