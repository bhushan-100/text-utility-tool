from jamspell import TSpellCorrector

corrector = TSpellCorrector()
corrector.LoadLangModel('en.bin')


def check_spelling(text):
    # split the text into words
    words = text.split()

    # create an empty list to store corrected words
    corrected = []

    # iterate over each word
    for word in words:
        # check if the word is misspelled
        fix = corrector.FixFragment(word)
        if fix != word:
            corrected.append(fix)
        else:
            corrected.append(word)

    # return the list of corrected words as a string
    return " ".join(corrected)
