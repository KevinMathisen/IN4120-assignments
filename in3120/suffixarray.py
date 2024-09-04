# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import sys
from bisect import bisect_left
from itertools import takewhile
from typing import Any, Dict, Iterator, Iterable, Tuple, List
from collections import Counter
from .corpus import Corpus
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .sieve import Sieve


class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__haystack: List[Tuple[int, str]] = []  # The (<document identifier>, <searchable content>) pairs.
        self.__suffixes: List[Tuple[int, int]] = []  # The sorted (<haystack index>, <start offset>) pairs.
        self.__build_suffix_array(fields)  # Construct the haystack and the suffix array itself.

    def __build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        for document in self.__corpus:
            document_content = ""

            # Normalize and combine all fields in the document
            for field in fields:
                document_content += (self.__normalize(document.get_field(field, "")) + " ")
            document_content = document_content.strip()

            # Insert document content into haystack as (<doc_id>, <content>)
            self.__haystack.append((document.get_document_id(), document_content))

            # Find and insert offsets for all suffixes as [doc_id, offset]   
            for term_position in self.__tokenizer.spans(document_content):
                self.__suffixes.append((len(self.__haystack)-1, term_position[0]))

        # Sort all suffixes based on the lexicographical order of the haystack/content
        self.__suffixes.sort(key=lambda suffix: self.__haystack[suffix[0]][1][suffix[1]:])

    def __normalize(self, buffer: str) -> str:
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """
        tokens = self.__tokenizer.tokens(self.__normalizer.canonicalize(buffer))
        return self.__normalizer.normalize(self.__tokenizer.join(tokens))

    def __binary_search(self, needle: str) -> int:
        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        prior to Python 3.10 due to how we represent the suffixes via (index, offset) tuples. Version 3.10
        added support for specifying a key.
        """
        # Define initial search area
        search_start_index = 0
        search_end_index = len(self.__suffixes)-1

        while True:
            # If we only have one suffix left to search, this is where the query should have been inserted
            if search_start_index >= search_end_index:
                return search_start_index
            
            # Find the suffix content of the middle suffix in our search area
            evaluating_index = int((search_end_index-search_start_index)/2) + search_start_index
            evaluating_content = self.__get_suffix_from_suffix_index(evaluating_index)

            # Check if the current suffix matches the query, or if we need to search lower or higher in the suffix array
            if needle == evaluating_content:
                return evaluating_index
            elif needle < evaluating_content:
                search_end_index = evaluating_index-1
            else:
                search_start_index = evaluating_index+1


    def evaluate(self, query: str, options: dict) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.

        The client can supply a dictionary of options that controls this query evaluation process: The maximum
        number of documents to return to the client is controlled via the "hit_count" (int) option.

        The results yielded back to the client are dictionaries having the keys "score" (int) and
        "document" (Document).
        """
        normalized_query = self.__normalize(query)
        if normalized_query == "":
            return iter([])

        # Find where we should start our search
        iterator_index = self.__binary_search(normalized_query)
        document_frequency = Counter("")

        # Check if the first match contains the query
        if self.__get_suffix_from_suffix_index(iterator_index).startswith(normalized_query):
            document_frequency[self.__get_doc_id_from_suffix_index(iterator_index)] += 1

        # Move down the array as long the predix match, counting the amount of times each document appear
        while iterator_index+1 < len(self.__suffixes) and self.__get_suffix_from_suffix_index(iterator_index+1).startswith(normalized_query):
            iterator_index += 1

            document_frequency[self.__get_doc_id_from_suffix_index(iterator_index)] += 1

        # Convert Counter to array of (<frequency>, <document_id>), compatible with Sieve
        documents_found = [(document_frequency[doc_id], doc_id) for doc_id in sorted(document_frequency)]

        # Cut off and rank documents based on their score, using the max hit_count when provided
        documents_sieve = Sieve(options["hit_count"] if options["hit_count"] else len(documents_found))
        documents_sieve.sift2(documents_found)

        # Return all documents within cutoff arranged by score
        for document in documents_sieve.winners():
            yield {"score": document[0], "document": self.__corpus.__getitem__(document[1])}

    def __get_suffix_from_suffix_index(self, suffix_index: int) -> str:
        return self.__haystack[self.__suffixes[suffix_index][0]][1][self.__suffixes[suffix_index][1]:]
    
    def __get_doc_id_from_suffix_index(self, suffix_index: int) -> str:
        return self.__haystack[self.__suffixes[suffix_index][0]][0]
    