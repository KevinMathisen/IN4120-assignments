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

        # Retrieve the posting lists for each terms in query, as (posting list, term)
        posting_lists = [(self.__inverted_index.get_postings_iterator(term), term) for term in sorted(terms) if self.__inverted_index.get_postings_iterator(term)]

        # Retrieve documents with highest score
        top_docs = self._soft_union(posting_lists, n, k, terms, ranker)

        # Yield the top documents score and document object from corpus
        yield from ({"score": doc[0], "document": self.__corpus.get_document(doc[1])} for doc in top_docs.winners())


    def _soft_union(self, posting_iterators: List[Iterator[Posting]], n: int, k: int, terms: Counter, ranker: Ranker) -> Sieve:
        """
        Goes through all posting lists at the same time, finding documents with at least n terms
        These documents have their relevancy caulcated with the ranker
        The documents are then ordered by using Sieve
        """
        top_docs = Sieve(k)
        # Start at head
        current_postings = [(next(posting_list_head, None), term) for (posting_list_head, term) in posting_iterators]

        # Remove all None elements from current_postings and corresponding iterators
        posting_lists_to_remove = [i for i, (posting, term) in enumerate(current_postings) if posting is None]
        for i in sorted(posting_lists_to_remove, reverse=True):
            posting_iterators.pop(i)
            current_postings.pop(i)

        # Go through all postings lists using document-at-a-time until no more matches possible      
        while len(posting_iterators) >= n:
            smallest_doc_ids = (-1, [])
            
            # Find the smallest docIDs of the iterators, saving their postings and terms
            for i, (posting, term) in enumerate(current_postings):
                if smallest_doc_ids[0] == -1 or posting.document_id < smallest_doc_ids[0]:
                    smallest_doc_ids = (posting.document_id, [(i, posting, term)])
                elif posting.document_id == smallest_doc_ids[0]:
                    smallest_doc_ids[1].append((i, posting, term))

            # If the smallest docID has postings over treshold n, calculate its score and sift through sieve
            if len(smallest_doc_ids[1]) >= n:
                ranker.reset(smallest_doc_ids[0])
                for (i, posting, term) in smallest_doc_ids[1]:
                    ranker.update(term, terms[term], posting)
                top_docs.sift(ranker.evaluate(), smallest_doc_ids[0])

            # Increment all smallest postings, removing them from posting lists when done
            posting_lists_to_remove = []
            for (i, posting, term) in smallest_doc_ids[1]:
                try:
                    current_postings[i] = (next(posting_iterators[i][0]), term)
                except StopIteration:
                    posting_lists_to_remove.append(i)

            for i in sorted(posting_lists_to_remove, reverse=True):
                    posting_iterators.pop(i)
                    current_postings.pop(i)

        return top_docs
    
    def _get_counter_terms(self, query: str) -> Counter:
        normalizer = SimpleNormalizer()
        tokenizer = SimpleTokenizer()
        tokens = (normalizer.normalize(t) for t in tokenizer.strings(normalizer.canonicalize(query)))
        return Counter(tokens)
        
