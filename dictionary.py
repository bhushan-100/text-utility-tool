import json
from difflib import get_close_matches

words_data = json.load(open("words.json"))


def word_meaning(word):

    # translate the word in lowercase, since all the words in json file is in lower case.
    word = word.lower()

    # check if word exists in words_data dict, If yes than return the value of that word by using d[key] method of dictionary,
    if word in words_data:
        return words_data[word][0]

    # tile() is a string method which converts the First letter of every word in uppercase, since some words have first letter in uppercase
    # so we will be checking it in below mentioned way as well.
    elif word.title() in words_data:
        return words_data[word.title()][0]

    elif word.upper() in words_data:
        return words_data[word.upper()][0]

    elif len(get_close_matches(word, words_data.keys())) > 0:
        similar_words_list = list(
            map(str, get_close_matches(word, words_data.keys())))

        similar = f"Did you mean '{similar_words_list[0]}' instead?"

        return f"{similar}\n{word_meaning(get_close_matches(word, words_data.keys())[0])}"

    else:
        return "Word doesn't exist."
