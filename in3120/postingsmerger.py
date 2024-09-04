# pylint: disable=missing-module-docstring

from typing import Iterator
from .posting import Posting


class PostingsMerger:
    """
    Utility class for merging posting lists.

    It is currently left unspecified what to do with the term frequency field
    in the returned postings when document identifiers overlap. Different
    approaches are possible, e.g., an arbitrary one of the two postings could
    be returned, or the posting having the smallest/largest term frequency, or
    a new one that produces an averaged value, or something else.

    Note that the result of merging posting lists is itself a posting list.
    Hence the merging methods can be combined to compute the result of more
    complex Boolean operations over posting lists.
    """

    @staticmethod
    def intersection(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple AND(A, B) of two posting
        lists A and B, given iterators over these.

        In set notation, this corresponds to computing the intersection
        D(A) ∩ D(B), where D(A) and D(B) are the sets of documents that
        appear in A and B: A posting appears once in the result if and
        only if the document referenced by the posting appears in both
        D(A) and D(B).

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)
        posting2 = next(iter2, None)

        # Yield posting with average of posting frequency when equal doc id
        while posting1 and posting2:
            if posting1.document_id == posting2.document_id:
                yield Posting(posting1.document_id, int(round((posting1.term_frequency+posting2.term_frequency)/2)))
                posting1 = next(iter1, None)
                posting2 = next(iter2, None)
            elif posting1.document_id < posting2.document_id:
                posting1 = next(iter1, None)
            else:
                posting2 = next(iter2, None)

    @staticmethod
    def union(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple OR(A, B) of two posting
        lists A and B, given iterators over these.

        In set notation, this corresponds to computing the union
        D(A) ∪ D(B), where D(A) and D(B) are the sets of documents that
        appear in A and B: A posting appears once in the result if and
        only if the document referenced by the posting appears in either
        D(A) or D(B).

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)
        posting2 = next(iter2, None)

        while posting1 or posting2:
            # If reached end of one iter, yield all remaining postings
            if not posting2:
                yield posting1
                posting1 = next(iter1, None)
            elif not posting1:
                yield posting2
                posting2 = next(iter1, None)

            # Yield average of two posting frequency when equal doc id
            elif posting1.document_id == posting2.document_id:
                yield Posting(posting1.document_id, int(round((posting1.term_frequency+posting2.term_frequency)/2)))
                posting1 = next(iter1, None)
                posting2 = next(iter2, None)

            # Yield smallest posting and iterate if not the same
            elif posting1.document_id < posting2.document_id:
                yield posting1
                posting1 = next(iter1, None)
            else:
                yield posting2
                posting2 = next(iter2, None)

    @staticmethod
    def difference(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple ANDNOT(A, B) of two posting
        lists A and B, given iterators over these.

        In set notation, this corresponds to computing the difference
        D(A) - D(B), where D(A) and D(B) are the sets of documents that
        appear in A and B: A posting appears once in the result if and
        only if the document referenced by the posting appears in D(A)
        but not in D(B).

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)
        posting2 = next(iter2, None)

        # Yield postings which only exist in iter 1
        while posting1 and posting2:
            if posting1.document_id == posting2.document_id:
                posting1 = next(iter1, None)
                posting2 = next(iter2, None)
            elif posting1.document_id < posting2.document_id:
                yield posting1
                posting1 = next(iter1, None)
            else:
                posting2 = next(iter2, None)

        # If iter1 has more postings, yield them
        while posting1:
            yield posting1
            posting1 = next(iter1, None)
