from chatgptaylor.vocabulary import Vocabulary


def test_init_vocabulary():
    """
    Test the vocabulary creation without initial tokens.
    """
    vocabulary = Vocabulary()

    assert len(vocabulary) == 1


def test_init_vocabulary_with_init_tokens():
    """
    Test the vocabulary creation with initial tokens.
    """
    init_tokens = ["extra1", "extra2", "extra3"]
    vocabulary = Vocabulary(init_tokens=init_tokens)

    assert len(vocabulary) == len(init_tokens) + 1
    assert len(vocabulary._vocabulary_transposed) == len(vocabulary)


def test_add_token():
    """
    Test add a token to the vocabulary.
    """
    vocabulary = Vocabulary()
    token = "token1"
    vocabulary.add_token(token)

    assert len(vocabulary) == 2
    assert vocabulary[token] == 1
    assert vocabulary._vocabulary_transposed[1] == token


def test_add_tokens():
    """
    Test add multiple tokens to the vocabulary.
    """
    vocabulary = Vocabulary()
    tokens = ["token1", "token2", "token3"]
    vocabulary.add_tokens(tokens)

    assert len(vocabulary) == len(tokens) + 1

    assert vocabulary[tokens[0]] == 1
    assert vocabulary._vocabulary_transposed[1] == tokens[0]

    assert vocabulary[tokens[1]] == 2
    assert vocabulary._vocabulary_transposed[2] == tokens[1]

    assert vocabulary[tokens[2]] == 3
    assert vocabulary._vocabulary_transposed[3] == tokens[2]


def test_add_tokens_enforce_sorted():
    """
    Test add multiple tokens to the vocabulary with enforce_sorted True.
    """
    vocabulary = Vocabulary()
    sorted_tokens = ["token1", "token2", "token3"]
    unsorted_tokens = ["token2", "token3", "token1"]
    vocabulary.add_tokens(unsorted_tokens, enforce_sorted=True)

    assert len(vocabulary) == len(sorted_tokens) + 1

    assert vocabulary[sorted_tokens[0]] == 1
    assert vocabulary._vocabulary_transposed[1] == sorted_tokens[0]

    assert vocabulary[sorted_tokens[1]] == 2
    assert vocabulary._vocabulary_transposed[2] == sorted_tokens[1]

    assert vocabulary[sorted_tokens[2]] == 3
    assert vocabulary._vocabulary_transposed[3] == sorted_tokens[2]


def test_add_repeated_tokens():
    """
    Test add multiple tokens to the vocabulary with enforce_sorted True.
    """
    vocabulary = Vocabulary()
    tokens = ["token1", "token1"]
    vocabulary.add_tokens(tokens, enforce_sorted=True)

    assert len(vocabulary) == len(set(tokens)) + 1

    assert vocabulary[tokens[0]] == 1
    assert vocabulary._vocabulary_transposed[1] == tokens[0]


def test_encode_sentence():
    """
    Test encode a sentence
    """
    tokens = ["token1", "token2", "token3", "token4", "token5"]
    vocabulary = Vocabulary()
    vocabulary.add_tokens(tokens)

    sentence = ["token3", "token2", "token3", "token4"]
    expected_encoded_sentence = [3, 2, 3, 4]

    encoded_sentence = vocabulary.encode(sentence)
    assert encoded_sentence == expected_encoded_sentence


def test_encode_batch():
    """
    Test encode a batch of sentences
    """
    tokens = ["token1", "token2", "token3", "token4", "token5"]
    vocabulary = Vocabulary()
    vocabulary.add_tokens(tokens)

    sentences = [
        ["token3", "token2", "token3", "token4"],
        ["token3", "token1", "token5"],
        ["token4", "token2", "token3", "token5"],
    ]
    expected_encoded_sentences = [
        [3, 2, 3, 4],
        [3, 1, 5],
        [4, 2, 3, 5]
    ]

    encoded_sentences = vocabulary.encode_batch(sentences)
    assert encoded_sentences == expected_encoded_sentences


def test_decode_sentence():
    """
    Test decode a sentence
    """
    tokens = ["token1", "token2", "token3", "token4", "token5"]
    vocabulary = Vocabulary()
    vocabulary.add_tokens(tokens)

    encoded_sentence = [3, 2, 3, 4]
    expected_decoded_sentence = ["token3", "token2", "token3", "token4"]

    decoded_sentence = vocabulary.decode(encoded_sentence)
    assert decoded_sentence == expected_decoded_sentence


def test_decode_batch():
    """
    Test decode a batch of sentences
    """
    tokens = ["token1", "token2", "token3", "token4", "token5"]
    vocabulary = Vocabulary()
    vocabulary.add_tokens(tokens)

    encoded_sentences = [
        [3, 2, 3, 4],
        [3, 1, 5],
        [4, 2, 3, 5]
    ]
    expected_decoded_sentences = [
        ["token3", "token2", "token3", "token4"],
        ["token3", "token1", "token5"],
        ["token4", "token2", "token3", "token5"],
    ]

    decoded_sentences = vocabulary.decode_batch(encoded_sentences)
    assert decoded_sentences == expected_decoded_sentences
