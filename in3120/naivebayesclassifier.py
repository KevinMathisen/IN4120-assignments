# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
from collections import Counter
from typing import Any, Dict, Iterable, Iterator
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus


class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier. For a detailed primer, see
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html.
    """

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str],
                 normalizer: Normalizer, tokenizer: Tokenizer):
        """
        Trains the classifier from the named fields in the documents in the
        given training set.
        """
        # Used for breaking the text up into discrete classification features.
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer

        # The vocabulary we've seen during training.
        self.__vocabulary = InMemoryDictionary()

        # Maps a category c to the logarithm of its prior probability,
        # i.e., c maps to log(Pr(c)).
        self.__priors: Dict[str, float] = {}

        # Maps a category c and a term t to the logarithm of its conditional probability,
        # i.e., (c, t) maps to log(Pr(t | c)).
        self.__conditionals: Dict[str, Dict[str, float]] = {}

        # Maps a category c to the denominator used when doing Laplace smoothing.
        self.__denominators: Dict[str, int] = {}

        # Train the classifier, i.e., estimate all probabilities.
        self.__compute_priors(training_set)
        self.__compute_vocabulary(training_set, fields)
        self.__compute_posteriors(training_set, fields)

    def __compute_priors(self, training_set) -> None:
        """
        Estimates all prior probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        total_documents = sum([documents.size() for documents in training_set.values()])

        # Calculate the prior for each class by divinding their document count with the total document count
        for (class_name, class_documents) in training_set.items():
            self.__priors[class_name] = math.log(class_documents.size() / total_documents)

    def __compute_vocabulary(self, training_set, fields) -> None:
        """
        Builds up the overall vocabulary as seen in the training set.
        """
        training_set = " ".join(
            [document.get_field(field, "") for class_documents in training_set.values() for document in class_documents for field in fields]
        )
        for term in self.__get_terms(training_set):
            self.__vocabulary.add_if_absent(term)

    def __compute_posteriors(self, training_set, fields) -> None:
        """
        Estimates all conditional probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        # Iterate through each class 
        for (class_name, class_documents) in training_set.items():
            text_for_class = " ".join([document.get_field(field, "") for document in class_documents for field in fields])
            
            terms = list(self.__get_terms(text_for_class))

            # Calculate the class denominator by adding the amount of terms in the class, and the amount of uniqe terms in the classifer (Because of Laplace smoothing)
            self.__denominators[class_name] = len(terms) + self.__vocabulary.size()

            term_occurances = Counter(terms)

            # Calculate the posteriors for each term by using the term occurance in the class, adding 1, and dividing on the demonitator for the class
            for (term, _) in self.__vocabulary:
                score = math.log((term_occurances[term]+1)/(self.__denominators[class_name]))
                self.__conditionals.setdefault(class_name, {})
                self.__conditionals[class_name][term] = score

    def __get_terms(self, buffer) -> Iterator[str]:
        """
        Processes the given text buffer and returns the sequence of normalized
        terms as they appear. Both the documents in the training set and the buffers
        we classify need to be identically processed.
        """
        tokens = self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))
        return (self.__normalizer.normalize(t) for t in tokens)

    def get_prior(self, category: str) -> float:
        """
        Given a category c, returns the category's prior log-probability log(Pr(c)).

        This is an internal detail having public visibility to facilitate testing.
        """
        return self.__priors[category]

    def get_posterior(self, category: str, term: str) -> float:
        """
        Given a category c and a term t, returns the posterior log-probability log(Pr(t | c)).

        This is an internal detail having public visibility to facilitate testing.
        """
        # Lookup in self.__conditionals
        if self.__vocabulary.get_term_id(term) == None:
            return 0.0
        
        return self.__conditionals[category][term]

    def classify(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.

        The results yielded back to the client are dictionaries having the keys "score" (float) and
        "category" (str).

        e.g. 
        dict("score": 1, "category": "eng")
        dict("score": 0, "category": no")
        """
        terms = list(self.__get_terms(buffer))

        scores = []
        for category in self.__priors.keys():
            # Calculate the score for each category by taking the sum of the category prior score and the term-category posterior scores
            category_score = self.get_prior(category)
            for term in terms:
                category_score += self.get_posterior(category, term)

            scores.append({"score": category_score, "category": category})
 
        # Sort the category scores based on their score in decending order
        yield from sorted(scores, key=lambda x: x["score"], reverse=True)
