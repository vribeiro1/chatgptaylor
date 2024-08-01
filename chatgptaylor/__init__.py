import os

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(BASEDIR, "resources")

TOKENIZER_VOCABULARY_FILEPATH = os.path.join(RESOURCES_DIR, "tokenizer", "vocabulary.pickle")
TOKENIZER_MERGES_FILEPATH = os.path.join(RESOURCES_DIR, "tokenizer", "merges.pickle")
