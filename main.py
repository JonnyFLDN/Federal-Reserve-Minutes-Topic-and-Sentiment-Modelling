import sys
import os
from utils.corpusutils import CorpusPreProcess, Document, Corpus
from utils.featureutils import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import pickle
from nltk.tokenize import WordPunctTokenizer
from nltk.data import LazyLoader
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import string
import json


def load_models(path):
    print("loading models")
    global transformer_model, transformer_tokenizer, tokenizer_settings
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(path, "transformer_model")
    )
    # Save LDA,vectorizer,topic_dictionary
    transformer_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(path, "transformer_tokenizer")
    )
    # save tokenizer settings
    # pickle.dump(tokenizer_settings,open(os.path.join(save_path,"tokenizer_settings.pkl"),"wb"))
    with open(os.path.join(path, "tokenizer_settings.pkl"), "rb") as settings:
        tokenizer_settings = pickle.load(settings)
    # save lda model
    global lda_model, vectorizer, lda_topic_dict
    with open(os.path.join(path, "lda_models.pkl"), "rb") as models:
        lda_model, vectorizer, lda_topic_dict = pickle.load(models)


def load_corpus(root, settings={}):
    """Load text into Corpus class"""

    word_tokenizer = WordPunctTokenizer()
    sent_tokenizer = LazyLoader("tokenizers/punkt/english.pickle")
    category_pattern = r"(\d{4})/*"
    file_extension = r"*.txt"
    file_pattern = r"(\d{8})/*"
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = (
        stopwords.words("english")
        + list(string.punctuation)
        + ["u", ".", "s", "--", "-", '."', ',"', ".)", ")-", '".', "—", "),"]
    )

    corpus_processor = CorpusPreProcess(
        root=root,
        file_extension=file_extension,
        category_pattern=category_pattern,
        file_pattern=file_pattern,
        word_tokenizer=word_tokenizer,
        sent_tokenizer=sent_tokenizer,
        stemmer=stemmer,
        lemmatizer=lemmatizer,
        stop_words=stop_words,
    )
    return corpus_processor


def load_features(corpus_processor):
    """Update corpus features"""

    corpus = corpus_processor.get_paras(flatten=True, stem=True)

    document_feat = FeatureProcessor(
        corpus,
        transformer_model=transformer_model,
        transformer_tokenizer=transformer_tokenizer,
        tokenizer_settings=tokenizer_settings,
        lda_model=lda_model,
        lda_vec=vectorizer,
        lda_topic_dict=lda_topic_dict,
        batch_size=30,
    )

    return document_feat.get_features()


def load_json(path, year):
    """Load pre-processed corpus"""
    file_name = "corpus_{}.json".format(year)
    with open(os.path.join(path, file_name), "r") as f:
        corpus_json = json.load(f)

        corpus = Corpus([Document(**i) for i in corpus_json["category"]])

    return corpus


path_text = r"US labour costs have risen sharply, contributing to the rapid climb of inflation as the Federal Reserve prepares to act forcefully to temper demand in the world's largest economy"
year = 2021
current_path = os.getcwd()
os.chdir(os.getcwd())

# Load pre-trained models
load_models(os.path.join(current_path, "Model"))

# Load path_text into Corpus class
if os.path.exists(path_text):
    ispath = True
    corpus_processor = load_corpus(path_text)
else:
    ispath = False
    temp_tuple = [("2022", "20220131", path_text)]
    corpus_processor = load_corpus(temp_tuple)

# extract features
corpus_features = load_features(corpus_processor)
json_path = os.path.join(current_path, "JSON")
compare_corpus = load_json(json_path, year)

# save comparison results
results = find_closest(corpus_features, compare_corpus)
output_path = os.path.join(current_path, "Output")
if not os.path.exists(output_path):
    os.mkdir(output_path)

for n, r in enumerate(results):
    r.to_csv(os.path.join(output_path, "output_{}".format(n)))
