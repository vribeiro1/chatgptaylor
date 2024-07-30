import os

from glob import glob

from chatgptaylor.tokenizer import Tokenizer


if __name__ == "__main__":
    basedir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(basedir, "resources")
    os.makedirs(resources_dir, exist_ok=True)
    tokenizer_dir = os.path.join(resources_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    vocabulary_filepath = os.path.join(tokenizer_dir, "vocabulary.pickle")
    merges_filepath = os.path.join(tokenizer_dir, "merges.pickle")

    datadir = os.path.join(basedir, "data")
    documents = sorted(glob(os.path.join(datadir, "Taylor Swift", "*.txt")))

    # tokenizer = Tokenizer()
    # tokenizer.train(documents)
    # tokenizer.dump(vocabulary_filepath, merges_filepath)

    tokenizer = Tokenizer.load(vocabulary_filepath, merges_filepath)

    for k, v in tokenizer.vocabulary.items():
        print(f"{k}    {v}")
    print("\n")

    input_english = "Hi! I'm ChatGPTaylor. An AI trained to generate pop songs inspired by Taylor Swift 😊"
    encoded = tokenizer.encode(input_english)
    decoded = tokenizer.decode(encoded)

    print("Original input :", input_english)
    print("Decoded :", decoded)

    input_portuguese = "Olá! Eu sou ChatGPTaylor. Uma IA treinada para gerar músicas pop inspirada pela Taylor Swift 😊"
    encoded = tokenizer.encode(input_portuguese)
    decoded = tokenizer.decode(encoded)

    print("Original input :", input_portuguese)
    print("Decoded :", decoded)
