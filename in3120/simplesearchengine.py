# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals

from collections import Counter
from typing import List, Iterator, Dict, Any
from .sieve import Sieve
from .ranker import Ranker
from .corpus import Corpus
from .invertedindex import InvertedIndex
from .posting import Posting
from .normalizer import SimpleNormalizer
from .tokenizer import SimpleTokenizer

class SimpleSearchEngine:
    """
    Realizes a simple query evaluator that efficiently performs N-of-M matching over an inverted index.
    I.e., if the query contains M unique query terms, each document in the result set should contain at
    least N of these m terms. For example, 2-of-3 matching over the query 'orange apple banana' would be
    logically equivalent to the following predicate:

       (orange AND apple) OR (orange AND banana) OR (apple AND banana)
       
    Note that N-of-M matching can be viewed as a type of "soft AND" evaluation, where the degree of match
    can be smoothly controlled to mimic either an OR evaluation (1-of-M), or an AND evaluation (M-of-M),
    or something in between.

    The evaluator uses the client-supplied ratio T = N/M as a parameter as specified by the client on a
    per query basis. For example, for the query 'john paul george ringo' we have M = 4 and a specified
    threshold of T = 0.7 would imply that at least 3 of the 4 query terms have to be present in a matching
    document.
    """

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self.__corpus = corpus
        self.__inverted_index = inverted_index

    def evaluate(self, query: str, options: Dict[str, Any], ranker: Ranker) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing N-out-of-M ranked retrieval. I.e., for a supplied query having M
        unique terms, a document is considered to be a match if it contains at least N <= M of those terms.

        The matching documents, if any, are ranked by the supplied ranker, and only the "best" matches are yielded
        back to the client as dictionaries having the keys "score" (float) and "document" (Document).

        The client can supply a dictionary of options that controls the query evaluation process: The value of
        N is inferred from the query via the "match_threshold" (float) option, and the maximum number of documents
        to return to the client is controlled via the "hit_count" (int) option.
        """
        # Get terms (as Counter) from query
        terms = self._get_counter_terms(query)
        # cauclate n (default m), and k (default 100)
        n = max(1, min(len(terms), int((options["match_threshold"] if options["match_threshold"] else 1) * len(terms))))
        k = options["hit_count"] if options["hit_count"] else 100 
        top_docs = Sieve(k)

        # Retrieve the posting lists from the inverted index for the terms in the query
        posting_lists = (self.__inverted_index.get_postings_iterator(term) for term in sorted(terms))
        # TODO: remove empty iterators
        # TODO: maybe have posting_lists be [(posting_iter, term), (), ...]

        # Go trough the all the postings lists at the same time        
        while len(posting_lists) >= n:
            smallest_doc_ids = (-1, []) # TODO: use infinite
            # check how many smallest
            for posting in posting_lists:
                if smallest_doc_ids[0] == -1 or posting.document_id < smallest_doc_ids[0]:
                    smallest_doc_ids = (posting.document_id, [posting]) # TODO: save term for each posting
                elif posting.document_id == smallest_doc_ids[0]:
                    smallest_doc_ids[1].append(posting)

            # if over treshold n, calculate/add to sieve
            if len(smallest_doc_ids[1]) >= n:
                ranker.reset(smallest_doc_ids[0])
                for posting in smallest_doc_ids[1]: # TODO: get the term for each posting
                    ranker.update(term, terms[term], posting)
                top_docs.sift(ranker.evaluate(), smallest_doc_ids[0])

            # increment all smallest
            for posting in smallest_doc_ids[1]:
                posting = next(posting, None)
            # if no more to increment, remove from postingslists
            # also, ensure that this will actually update the posting in the posting_lists

            # TODO: reset smallest docids (maybe, might be able to keep for performance)

        # yield sieve
        yield from ({"score": doc[0], "document": self.__corpus.get_document(doc[1])} for doc in top_docs.winners())


    def _soft_union(self, posting_lists: List[Iterator[Posting]], n: int) -> Iterator[Posting]:
        # Should go through all posting lists at the same time
        # The pointer(s) to the posting with the least ID should be incremented
        # When pointers with same ID is over n (treshold), 
        #       caclulate its score by going through all terms, and add it to the sieve
        # when there are fewer postings lists left to iterate than n, exit the function

        yield iter()
    
    def _get_counter_terms(self, query: str) -> Counter:
        tokens = (SimpleNormalizer.normalize(t) for t in SimpleTokenizer.strings(SimpleNormalizer.canonicalize(query)))
        return Counter(tokens)
        
