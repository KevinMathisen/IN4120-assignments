# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

from typing import Iterator, Dict, Any, List, Tuple
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications and some minor
    NLP extensions. The running time of this algorithm is virtually independent of the size of the dictionary,
    and linear in the length of the buffer we are searching in.

    The tokenizer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    def __init__(self, trie: Trie, normalizer: Normalizer, tokenizer: Tokenizer):
        self.__trie = trie
        self.__normalizer = normalizer  # The same as was used for trie building.
        self.__tokenizer = tokenizer  # The same as was used for trie building.

        self.__output: Dict[Trie, List[str]] = {}
        self.__failure: Dict[Trie, Trie] = {}   # key: node to find failure for, value: where to move if key has failure

        self.__build_output()
        self.__build_failure()

    def scan(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Scans the given buffer and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        The matches, if any, are yielded back to the client as dictionaries having the keys "match" (str),
        "surface" (str), "meta" (Optional[Any]), and "span" (Tuple[int, int]). Note that "match" refers to
        the matching dictionary entry, "surface" refers to the content of the input buffer that triggered the
        match (the surface form), and "span" refers to the exact location in the input buffer where the surface
        form is found. Depending on the normalizer that is used, "match" and "surface" may or may not differ.

        A space-normalized version of the surface form is emitted as "surface", for convenience. Clients
        that require an exact surface form that is not space-normalized can easily reconstruct the desired
        string using the emitted "span" value.

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and more.
        """
        # space_normalized_buffer = self.__tokenizer.join(self.__tokenizer.tokens(self.__normalizer.canonicalize(buffer)))    # Used for outputting surface
        normalized_buffer = self._get_buffer_normalized(buffer)                                                             # Used for matching with trie  
        org_term_positions = self._get_normalized_span_to_span_mapping(self.__tokenizer.tokens(buffer))                          # Used for finding buffer span values

        node = self.__trie

        for buffer_symbol_index in range(0, len(normalized_buffer)):
            buffer_symbol = normalized_buffer[buffer_symbol_index]

            # Find node containing input symbol, by using fail() to iterate through new nodes
            while node.child(buffer_symbol) == None and node != self.__trie:
                node = self.__failure[node]
            # If no matching node found, continue to next symbol in buffer
            if node == self.__trie and node.child(buffer_symbol) == None:
                continue

            # move to node matching the input symbol
            node = node.child(buffer_symbol)

            # Check if output found
            outputs = self.__output.get(node, [])
            for match in outputs:
                match = self._generate_output(match, buffer_symbol_index, buffer, org_term_positions, node.get_meta())
                if match:
                    yield match


    def _generate_output(self, match: str, match_position: int, buffer: str, org_term_positions: int, meta: Any) -> Dict[str, Any]:
        # Retrieve positions needed for generating output
        start_pos_buffer = org_term_positions.get((match_position - len(match) + 1, True), -1)
        end_pos_buffer = org_term_positions.get((match_position+1, False), -1)

        surface = self.__tokenizer.join(self.__tokenizer.tokens(self.__normalizer.canonicalize(buffer[start_pos_buffer:end_pos_buffer])))

        # Check if match starts and ends at terms
        if start_pos_buffer == -1 or end_pos_buffer == -1:
            return None

        return {"match": match, "surface": surface, 
                       "meta": meta, "span": (start_pos_buffer, end_pos_buffer)}

    def __build_output(self):
        # Builds the output by saving where each term ends in the trie, and their value

        queue: List[Tuple[Trie, List[str]]] = []
        node: Trie = self.__trie # root
        current_term: List[str] = []

        queue.append((self.__trie, []))

        while len(queue) != 0:
            node, current_term = queue.pop(0)

            # If node is end of a term, save it to output
            if node.is_final():
                self.__output[node] = ["".join(current_term)]

            # Add next characters in queue if any
            for symbol in node.transitions():
                queue.append((node.child(symbol), current_term+[symbol]))        

    def __build_failure(self):
        # Builds a failure tree by saving where the scan should move if a failure occurs during scan
        queue: List[Trie] = []

        # Point all nodes with depth 1 to root
        for symbol in self.__trie.transitions():
            queue.append(self.__trie.child(symbol))
            self.__failure[self.__trie.child(symbol)] = self.__trie
        
        # Iterate through all nodes, one layer at a time
        while len(queue) != 0:
            node = queue.pop(0)

            for symbol in node.transitions():
                queue.append(node.child(symbol))
                
                # Find a node to point to in case of failure by iterating through the fail tree
                #   until we find another node which points to the same symbol, or we reach the root
                fail_node = self.__failure[node]
                while fail_node.child(symbol) == None and fail_node != self.__trie:
                    fail_node = self.__failure[fail_node]

                # Save the found node to point to in case of failure (root if none found)
                if fail_node == self.__trie and fail_node.child(symbol) == None:
                    self.__failure[node.child(symbol)] = self.__trie
                else:
                    self.__failure[node.child(symbol)] = fail_node.child(symbol)
                
                # Update the output of the node to include the fail nodes output if any
                if fail_node.child(symbol) in self.__output:
                    self.__output.setdefault(node.child(symbol), []).extend(self.__output[fail_node.child(symbol)])

    def _get_normalized_span_to_span_mapping(self, tokens: Iterator[Tuple[str, Tuple[int, int]]]) -> Dict[int, Tuple[int, bool]]:
        # this function is for mapping tokens from the original buffer to their equivalent in the normalized buffer
        # if achieves this by going through all original tokens,  and calculating the position these are placed in the final normalised buffer
        #   by using their length (after normalisation) 
        
        span_mapping = {}
        normalised_term_start_pos = 0
        end_pos_last_org_term = 0
        length_last_org_term = 0
        for org_term in tokens:
            # Calculate if there was space between this token and the last one
            space = 0 if org_term[1][0] - end_pos_last_org_term == 0 else 1
            space = 0 if end_pos_last_org_term == 0 else space
            normalised_term_start_pos += length_last_org_term+space # Jump to place of next term in normalised buffer
            end_pos_last_org_term = org_term[1][1]

            # Cant assume that term length after normalization is same as before
            length_org_term = len(self.__normalizer.normalize(self.__normalizer.canonicalize(org_term[0])))
            span_mapping[(normalised_term_start_pos, True)] = org_term[1][0] # Save start position of term to the place it appears in normalised buffer
            span_mapping[(normalised_term_start_pos+length_org_term, False)] = org_term[1][1] # Save end position of term to the place it appear in normalised buffer 
            length_last_org_term = length_org_term

        return span_mapping

    def _get_buffer_normalized(self, buffer: str) -> str:
        tokens = self.__tokenizer.tokens(self.__normalizer.canonicalize(buffer))
        return self.__tokenizer.join((self.__normalizer.normalize(token), _) for token, _ in tokens)