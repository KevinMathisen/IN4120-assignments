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
    """

    @staticmethod
    def intersection(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple AND(A, B) of two posting
        lists A and B, given iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)
        posting2 = next(iter2, None)

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

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)
        posting2 = next(iter2, None)

        while posting1 and posting2:
            if posting1.document_id == posting2.document_id:
                yield Posting(posting1.document_id, int(round((posting1.term_frequency+posting2.term_frequency)/2)))
                posting1 = next(iter1, None)
                posting2 = next(iter2, None)
            elif posting1.document_id < posting2.document_id:
                yield posting1
                posting1 = next(iter1, None)
            else:
                yield posting2
                posting2 = next(iter2, None)

        # If either iter1 or iter2 has more postings, yield them
        while posting1:
            yield posting1
            posting1 = next(iter1, None)

        while posting2:
            yield posting2
            posting2 = next(iter2, None)

    @staticmethod
    def difference(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple ANDNOT(A, B) of two posting
        lists A and B, given iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        posting1 = next(iter1, None)
        posting2 = next(iter2, None)

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
