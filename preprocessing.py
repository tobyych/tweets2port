import os
import re
import json
from nltk.corpus import stopwords
from textblob import TextBlob, Word
from nltk.stem import PorterStemmer
import datetime
import shutil


class Preprocessor:
    def __init__(self):
        pass

    def clean(self, s):
        s = self.LowerCase(s)
        s = self.RemoveURL(s)
        s = self.RemoveMention(s)
        s = self.RemoveEmoticons(s)
        s = self.RemoveEmoji(s)
        s = self.RemovePunc(s)
        s = self.RemoveNumbers(s)
        s = self.RemoveStopWords(s)
        s = self.Stemming(s)
        return self.Tokenisation(s)

    def LowerCase(self, s):
        return " ".join([x.lower() for x in s.split()])

    def RemoveNumbers(self, s):
        return re.sub("\d", "", s)

    def RemovePunc(self, s):
        return re.sub("[^A-Za-z0-9_\s]+", "", s)

    def RemoveStopWords(self, s):
        stop = stopwords.words("english")
        return " ".join([x for x in s.split() if x not in stop])

    def RemoveURL(self, s):
        URL_PATTERN = re.compile(
            r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'
        )
        return URL_PATTERN.sub("", s)

    def RemoveEmoji(self, s):
        return emoji_pattern.sub("", s)

    def RemoveEmoticons(self, s):
        return " ".join([x for x in s.split() if x not in emoticons])

    def ReplaceHashtag(self, s):
        HASHTAG_PATTERN = re.compile(r"#\w*")
        return HASHTAG_PATTERN.sub("$HASHTAG$", s)

    def RemoveMention(self, s):
        MENTION_PATTERN = re.compile(r"@\w*")
        return MENTION_PATTERN.sub("", s)

    def RemoveReservedWords(self, s):
        RESERVED_WORDS_PATTERN = re.compile(r"^(RT|FAV)")
        return RESERVED_WORDS_PATTERN.sub("", s)

    def RemoveCommonWords(self, s):
        pass

    def RemoveRareWords(self, s):
        pass

    def Tokenisation(self, s, spellcheck=False):
        if spellcheck:
            return TextBlob(s).correct().words
        return TextBlob(s).words

    def Stemming(self, s):
        stemmer = PorterStemmer()
        return " ".join([stemmer.stem(word) for word in s.split()])

    def Lemmatisation(self, s):
        return " ".join([Word(word).lemmatize() for word in s.split()])


emoticons_happy = set(
    [
        ":-)",
        ":)",
        ";)",
        ":o)",
        ":]",
        ":3",
        ":c)",
        ":>",
        "=]",
        "8)",
        "=)",
        ":}",
        ":^)",
        ":-D",
        ":D",
        "8-D",
        "8D",
        "x-D",
        "xD",
        "X-D",
        "XD",
        "=-D",
        "=D",
        "=-3",
        "=3",
        ":-))",
        ":'-)",
        ":')",
        ":*",
        ":^*",
        ">:P",
        ":-P",
        ":P",
        "X-P",
        "x-p",
        "xp",
        "XP",
        ":-p",
        ":p",
        "=p",
        ":-b",
        ":b",
        ">:)",
        ">;)",
        ">:-)",
        "<3",
    ]
)

emoticons_sad = set(
    [
        ":L",
        ":-/",
        ">:/",
        ":S",
        ">:[",
        ":@",
        ":-(",
        ":[",
        ":-||",
        "=L",
        ":<",
        ":-[",
        ":-<",
        "=\\",
        "=/",
        ">:(",
        ":(",
        ">.<",
        ":'-(",
        ":'(",
        ":\\",
        ":-c",
        ":c",
        ":{",
        ">:\\",
        ";(",
    ]
)

emoticons = emoticons_happy.union(emoticons_sad)

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def prepocess():
    p = Preprocessor()
    datapath = "../data/stocknet-dataset/tweet/raw"
    folderpath = [x[0] for x in os.walk(datapath)][1:]
    for folder in folderpath:
        print(f"current folder: {folder}")
        filelist = os.listdir(folder)
        filelist.sort()
        for filename in filelist:
            raw_text = list()
            with open(os.path.join(folder, filename), "r", encoding="ascii") as f:
                idx = 0
                for line in f:
                    line = json.loads(line)
                    raw_text.append(line["text"])

            clean_text_list = list()
            for idx, text in enumerate(raw_text):
                clean_text_list.append((idx, p.clean(text)))
            clean_text = dict(clean_text_list)

            ofpath = "../data/stocknet-dataset/tweet/new_processed/" + os.path.basename(
                os.path.normpath(folder)
            )
            if not os.path.exists(ofpath):
                os.makedirs(ofpath)
            ofname = filename
            with open(ofpath + "/" + ofname + ".json", "w") as f:
                f.write(json.dumps(clean_text))


stock_universe = [
    "VZ",
    "AMZN",
    "CAT",
    "AAPL",
    "PFE",
    "CELG",
    "GOOG",
    "WMT",
    "CVX",
    "INTC",
    "HD",
    "WFC",
    "MRK",
    "JNJ",
    "BABA",
    "BAC",
    "FB",
    "MSFT",
    "ABBV",
    "MCD",
    "T",
    "PG",
    "XOM",
    "BA",
    "GE",
    "AMGN",
    "DIS",
    "CSCO",
    "KO",
    "C",
    "D",
]


def delete_unused_files(
    datapath="../data/stocknet-dataset/tweet/new_processed/", filelist=stock_universe
):
    folderpath = [x[0] for x in os.walk(datapath)][1:]
    for folder in folderpath:
        if not os.path.basename(os.path.normpath(folder)) in filelist:
            shutil.rmtree(folder)


def add_null_files():
    null_dict = {0: []}
    datapath = "../data/stocknet-dataset/tweet/new_processed/"
    folderpath = [x[0] for x in os.walk(datapath)][1:]
    for folder in folderpath:
        date = datetime.date(2014, 1, 1)
        while date != datetime.date(2016, 3, 31):
            filepath = os.path.join(folder, date.strftime("%Y-%m-%d"))
            if not os.path.exists(filepath + ".json"):
                with open(filepath + ".json", "w+") as f:
                    f.write(json.dumps(null_dict))
            date += datetime.timedelta(1)


# prepocess()
delete_unused_files()
add_null_files()
