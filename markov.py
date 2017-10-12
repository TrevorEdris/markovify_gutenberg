import argparse
import random
import re
import requests

from string import (
    ascii_lowercase,
    ascii_uppercase,
)

import sys

NORMAL = '\033[00m'
RED = '\033[31m'
GREEN = '\033[92m'
BLUE = '\033[96m'

MAX_KEY_LENGTH = 10
MAX_OVERLAP_RATIO = 0.5
MAX_OVERLAP_TOTAL = 15
BEGIN = '__BEGIN__'
END = '__END__'

rejoined_text = ''
rejoined_text_lower = ''

abbr_capped = "|".join([
    "ala|ariz|ark|calif|colo|conn|del|fla|ga|ill|ind|kan|ky|la|md|mass|mich|minn|miss|mo|mont|neb|nev|okla|ore|pa|tenn|vt|va|wash|wis|wyo",  # States
    "u.s",
    "mr|ms|mrs|msr|dr|gov|pres|sen|sens|rep|reps|prof|gen|messrs|col|sr|jf|sgt|mgr|fr|rev|jr|snr|atty|supt",  # Titles
    "ave|blvd|st|rd|hwy",  # Streets
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec",  # Months
    "|".join(ascii_lowercase)  # Initials
]).split("|")

abbr_lowercase = "etc|v|vs|viz|al|pct"


def red(text):
    return f'{RED}{text}{NORMAL}'


def green(text):
    return f'{GREEN}{text}{NORMAL}'


def blue(text):
    return f'{BLUE}{text}{NORMAL}'


def is_abbreviation(dotted_word):
    """
    Checks if the dotted_word is a known abbreviation

    :param dotted_word: Word to check
    :returns: Boolean indicating if the word is a known abbreviation
    """
    clipped = dotted_word[:-1]
    if clipped[0] in ascii_uppercase:
        if clipped.lower() in abbr_capped:
            return True
        return False
    else:
        if clipped in abbr_lowercase:
            return True
        return False


def is_sentence_ender(word):
    """
    Checks if the word is the last word in a sentence

    :param word: Word to check
    :returns: Boolean indicating if the word is the end of a sentence
    """
    if word[-1] in '?!':
        return True
    if len(re.sub(r"[^A-Z]", "", word)) > 1:
        return True
    if word[-1] == '.' and not is_abbreviation(word):
        return True
    return False


def split_into_sentences(text):
    """
    Splits the text into a list of sentences, where each sentence
    is a list of words

    :param text: The text to split up
    :returns: List of lists
    """
    global rejoined_text
    global rejoined_text_lower
    potential_end_pattern = re.compile(r''.join([
        r"([\w\.'’&\]\)]+[\.\?!])",  # A word that ends with punctuation
        r"([‘’“”'\"\)\]]*)",         # A word followed by optional quote/parens/etc
        r"(\s+(?![a-z\-–—]))",       # A word followed by whitespace + non-(lowercase or dash)
        ]), re.U)
    dot_iter = re.finditer(potential_end_pattern, text)
    end_indices = [(x.start() + len(x.group(1)) + len(x.group(2)))
                   for x in dot_iter
                   if is_sentence_ender(x.group(1))]
    spans = zip([None] + end_indices, end_indices + [None])
    sentences = [text[start:end].strip() for start, end in spans]
    sentences = [f'{sentence} {END}' for sentence in sentences]
    sentences = [[word.strip() for word in sentence.split(' ') if word] for sentence in sentences]
    rejoined_text = '\n'.join([' '.join([word for word in sentence]) for sentence in sentences])
    rejoined_text_lower = rejoined_text.lower()
    return sentences


def sanitize(word):
    """
    Removes whitespace and parenthetical characters while
    keeping all chars in to_keep

    :param word: The text to sanitize
    :returns: Sanitized text
    """
    to_replace = '\n\t\r()<>\'\"'
    to_keep = '.!?,i:;@#$%^&*_+= '

    # Build list of characters to replace
    for char in word:
        if not char.isalnum() and char not in to_replace and char not in to_keep:
            to_replace += char
    for char in to_replace:
        word = word.replace(char, ' ')
    return word or ''


def get_book_text_url(url):
    """
    Retrieves the text of the book, provided the url

    :param url: The direct url to the utf-8 text file of the book
    :returns: Full text of the book
    """
    text = requests.get(url).text.lower()
    text_split = text.split(' ')
    text = ' '.join(sanitize(word) for word in text_split if word and word != ' ')
    return text


def get_book_text_file(fname):
    """
    Retrieves the text of the book, given the filename

    :param fname: Path to the text file of the book
    :returns: Full text of the book
    """
    with open(fname) as fp:
        text = '\n'.join(fp.readlines())
    return sanitize(text)


def build_markov_chain_from_sentences(sentences, key_length, chain=None):
    """
    Builds a markov chain in the form of a dict based off the
    input sentences

    :param sentences: List of sentences, where each sentence is a list of words
    :param key_length: Number of words to use in the chain
    :param chain: Existing chain or None.
                  Multiple chains can be combined
        EXAMPLE:
            build_markov_chain_from_sentences(moby_dick_sentences, 2, build_markov_chain_from_sentences(WotW_sentences, 2))
    """
    if chain is None:
        chain = {}

    # Limit the key length since at some point, original sentence
    # generation will be impossible
    if key_length > MAX_KEY_LENGTH:
        key_length = MAX_KEY_LENGTH

    for sentence in sentences:

        index = key_length
        _begin = True
        for i in range(len(sentence) - key_length):

            # Build list of sentence[index]'s previous key_length words
            # to use as a key for the dictionary
            keys = [sentence[index-i] for i in range(key_length, 0, -1)]
            key = ' '.join(keys)

            # Keep track of what words begin a sentence
            if _begin:
                if BEGIN in chain:
                    chain[BEGIN].append(key)
                else:
                    chain[BEGIN] = [key]
                _begin = False

            if key in chain:
                chain[key].append(sentence[index])
            else:
                chain[key] = [sentence[index]]
            index += 1

    return chain


def build_random_sentence(chain, key_length, msg_len=25, tries=10):
    """
    Attempts to generate a sentence based off the chain

    :param chain: The markov chain to use
    :param key_length: The number of words in the chain's keys
        * NOTE: Needs to be the same as what was used for build_markov_chain_from_sentences
    :param msg_len: Maximum number of words in the sentence
    :param tries: Maximum number of attempts to generate an original sentence
    """

    for i in range(tries):

        # Get a key from the words that begin a sentence
        words = random.choice(chain[BEGIN]).split(' ')
        sentence = words[0].capitalize()
        if key_length > 1:
            sentence += ' ' + ' '.join(words[1:])

        # Generate a maximum of msg_len words for the sentence
        invalid = False
        for i in range(msg_len - key_length):
            try:
                next_word = random.choice(chain[' '.join(words)])
                if next_word == END:
                    if test_generated_sentence(sentence.split(' ')):
                        return green(sentence)
                    else:
                        invalid = True
                        break
                sentence += ' ' + next_word
                del words[0]
                words.append(next_word)
            except KeyError:
                # Print something to let user know an error occured
                print('t(\'-\')t', end='')

        # If here, reached msg_len OR invalid sentence
        # Make sure sentence ends with punctuation
        if not invalid:
            if not sentence[-1] in '.?!':
                sentence += random.choice(list('.?!'))
            if test_generated_sentence(sentence.split(' ')):
                return green(sentence)

    return red('UNABLE TO GENERATE ORIGINAL SENTENCE')


def test_generated_sentence(words):
    """
    Checks if the generated tweet was original or not

    :param words: List of words in the sentence
    :returns: Boolean to indicate if the sentence is original or not
    """

    # If too many words overlap with a sentence
    # direct from the text, reject that sentence.
    overlap_ratio = int(round(MAX_OVERLAP_RATIO * len(words)))
    overlap_max = min(MAX_OVERLAP_TOTAL, overlap_ratio)
    overlap_over = overlap_max + 1
    gram_count = max((len(words) - overlap_max), 1)
    grams = [words[i:i+overlap_over] for i in range(gram_count)]
    for g in grams:
        gram_joined = ' '.join(g)
        if gram_joined.lower() in rejoined_text_lower:
            return False
    return True


def get_book_title(sentences):
    """
    Attempts to locate the title of the book given the
    sentences in the book. Generally, on gutenberg.org, the
    book title is preceeded by 'Title:' and followed by 'Author:'
    in the same sentence.

    :param sentences: All sentences in the book
    """
    title = []
    for sentence in sentences:
        title_found = False
        for word in sentence:
            if word == 'Author:':
                return ' '.join(title)
            if title_found:
                title.append(word)
            if word == 'Title:':
                title_found = True


def get_word_frequency(text):
    """
    Builds a {word: frequency} dict based off the text

    :param text: The text to build the frequency dict for
    :returns: Generated frequency dict
    """
    freqs = {}
    book_words = text.split(' ')
    for word in book_words:
        word = sanitize(word)
        if not word:
            continue
        if word in freqs:
            freqs[word] += 1
        else:
            freqs[word] = 1
    return freqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Generates a random block of text by building a Markov Chain
    generated from the text of a book''')
    parser.add_argument('-u', '--book_url', type=str,
            help='Direct URL of the book')
    parser.add_argument('-i', '--input_file', type=str,
            help='Path to text file')
    parser.add_argument('-o', '--output', type=str,
            help='Path to output file')
    parser.add_argument('-k', '--key_length', type=int,
            help='Number of words to use as key in the chain, max of 10')
    parser.add_argument('-m', '--message_length', type=int,
            help='Number of words in the generated message')
    parser.add_argument('-v', '--verbose', action='store_const', const=True, default=False,
            help='Verbose output, print generated message')
    parser.add_argument('-s', '--sentences', type=int,
            help='Number of random sentences to generate')
    args = parser.parse_args()

    if not args.book_url and not args.input_file:
        sys.stderr.write('Must specify a url or a file')
        sys.exit(1)
    elif args.book_url and args.input_file:
        sys.stderr.write('Specify only one of: [book_url, input_file]')
        sys.exit(1)
    elif args.book_url:
        text = get_book_text_url(args.book_url)
    elif args.input_file:
        text = get_book_text_file(args.input_file)

    if not args.key_length:
        args.key_length = 1

    if not args.output:
        args.output = 'output/markov_results.txt'

    if not args.message_length:
        args.message_length = 100

    if not args.sentences:
        args.sentences = 1

    sentences = split_into_sentences(text)

    book_title = get_book_title(sentences)
    s = f' {book_title} --> key_len: {args.key_length} '
    title = f'{s:=^80}'

    chain = build_markov_chain_from_sentences(sentences, args.key_length)
    full_msg = '\n\n'.join([build_random_sentence(chain, args.key_length) for i in range(args.sentences)])
    print(blue('\n' + title + '\n'))
    print(full_msg)

    with open(args.output, 'w') as fp:
        fp.write(blue(title) + '\n')
        fp.write(full_msg + '\n')
