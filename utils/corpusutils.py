import re
import os
from pathlib import Path
from nltk import pos_tag as pos_tag
import io
import json
from nltk.corpus import wordnet


def custom_get_pos(first_tag):
    """Custom lemmatizer get_pos"""

    if first_tag.startswith("J"):
        return wordnet.ADJ
    elif first_tag.startswith("V"):
        return wordnet.VERB
    elif first_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class Document(object):
    """
    Document class stores text information
    e.g. Document(tokens = ['this,'text','is',tokenized'])

    """

    def __init__(
        self,
        category_id,
        file_id,
        raw,
        tokens,
        lemma,
        stem,
        structure="sentence",
        features={},
    ):

        self.category_id = category_id
        self.file_id = file_id
        self.raw = raw
        self.tokens = tokens
        self.stem = stem
        self.lemma = lemma
        self.structure = structure
        self.features = features

    def __str__(self):
        return str(self.tokens)

    def __repr__(self):
        return str(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    def __len__(self):
        return len(self.tokens)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class Corpus(object):
    """
    Corpus class takes in a list of Document instances
    allows for corpus level analysis and export of json

    """

    def __init__(self, documents=[]):
        # Corpus takes in a list of documents
        if not isinstance(documents, list):
            raise ValueError("documents must be in a form of a list")
        # documents must be a list of Document instances
        if not all(self._check_valid_doc(d) for d in documents):
            if not documents == []:
                self._doc_type_exception()

        self.documents = documents

    def append(self, document):
        if not self._check_valid_doc(document):
            self._doc_type_exception()

        self.documents.append(document)

    def _check_valid_doc(self, document):
        if document.__class__.__name__ != Document.__name__:
            return False
        else:
            return True

    def _doc_type_exception(self):
        raise TypeError("document must be a Document instance")

    def __repr__(self):
        return str(self.documents)

    def __add__(self, other_corpus):
        return self.__class__(self.documents + other_corpus.documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    def extract_features(self, key, return_generator=False):
        if return_generator:
            return (d.features[key] for d in self.documents)
        return [d.features[key] for d in self.documents]

    def to_json(self, save_path, by_category=False):

        json_name = "corpus_{}.json"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if by_category:
            category_dict = {}

            for d in self.documents:
                if d.category_id in category_dict:
                    category_dict[d.category_id].append(d.__dict__)
                else:
                    category_dict[d.category_id] = [d.__dict__]

            for k, v in category_dict.items():
                joint_dict = {}
                joint_dict["category"] = v
                json_path = os.path.join(save_path, json_name.format(k))
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(joint_dict, f, sort_keys=True, indent=4)
        else:

            joint_dict = {}
            doc_dict = [d.__dict__ for d in self.documents]
            joint_dict["category"] = doc_dict

            json_path = os.path.join(save_path, json_name.format("all"))

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(joint_dict, f, sort_keys=True, indent=4)


class CorpusPreProcess(object):
    """
    CorpusPreProcess streams in .txt files or strings for pre-processing
    performs tokenization/stemming/lemmatization
    at a word,sentence and paragraph level.

    returns populated Document object
    e.g. [Document(structure='paragraph'),Document(structure='paragraph')]

    Methods
    --------
    get_file_ids : return file ids
    get_category_ids : return category ids
    _read_tuples : Read in tuples of (category_id,file_id,"text") when root!=path
    _read_paths : Read in text from path
    _load_objects : take in paths or tuples and load in string data
    _stem : Read tokens and return stemmed tokens
    _lemmatize : Read tokens and return lemmatized tokens
    get_words : return word tokens
    get_sents : return sentence-word tokens
    get_paras : return paragraph-word tokens
    truncate_text : open source files and truncate via regex
    read_block : read stream of text, output paragraph block


    """

    def __init__(
        self,
        root,
        file_extension,
        category_pattern,
        file_pattern,
        word_tokenizer,
        sent_tokenizer,
        lemmatizer=None,
        stemmer=None,
        block_reader=None,
        stop_words=[],
        encoding="utf-8",
    ):
        """

        :param root: path or list of tuples [(category_id,file_id,"text")]
        :param file_extension: extension to use with root, to find files
        :param category_pattern: regex pattern to extract category from file name
        :param file_pattern: regex pattern to extract file id from file name
        :param word_tokenizer: pre-defined work tokenizer
        :param sent_tokenizer: pre-defined sentence tokenizer
        :param lemmatizer: pre-defined lemmatizer
        :param stemmer: pre-defined stemmer
        :param block_reader: return paragraph block from text stream
        :param stop_words: pre-defined stop_words
        :param encoding:

        :returns: list of Document objects

        """

        self.file_extension = file_extension
        self.file_pattern = file_pattern
        self.category_pattern = category_pattern
        self.word_tokenizer = word_tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.encoding = encoding
        self.lemmatizer = lemmatizer
        self.stemmer = stemmer
        self.stop_words = stop_words
        self.file_ids = None

        self.Document = Document
        self.Corpus = Corpus

        if block_reader:
            self.block_reader = block_reader

        self.root_paths = None
        # check if root is a path or a list of tuples
        if isinstance(root, str):
            if os.path.exists(root):
                self.root = Path(root)
                # store init_root for truncation method
                self.init_root = self.root
                self.isroot = True
        elif isinstance(root, list):
            if not all(isinstance(i, tuple) * (len(i) == 3) for i in root):
                raise ValueError(
                    "Input text needs nested tuples [(category_id, file_id,document)]"
                )
            else:
                self.isroot = False
                self.root = root
        else:
            raise ValueError(
                "Root needs an existing address or a nested list of tuples"
            )

        # load in paths/lists into a dictionary
        if self.isroot:
            self._read_paths()
        else:
            self._read_tuples()

        self._file_ids = list(self.file_root_paths.keys())
        self._category_ids = list(self.cat_root_paths.keys())

    def get_file_ids(self, category_id=None):
        """Return file ids"""
        if category_id:
            return list(self.cat_root_paths[category_id].keys())
        else:
            return list(self.file_root_paths.keys())

    def get_category_ids(self):
        """Return category ids"""
        return self._category_ids

    def _read_tuples(self):
        """Populates dictionaries with root content
        Values are str format
        """
        cat_root_paths = {}
        file_root_paths = {}

        for category_id, file_id, item in self.root:
            file_root_paths[file_id] = (category_id, file_id, item)
            d = {file_id: (category_id, file_id, item)}

            if category_id in cat_root_paths:
                cat_root_paths[category_id].update(d)
            else:
                cat_root_paths[category_id] = d

        self.cat_root_paths = cat_root_paths
        self.file_root_paths = file_root_paths

    def _read_paths(self):
        """Populates dictionaries with root content
        Values are paths
        """
        paths = sorted(self.root.glob(self.file_extension))

        cat_root_paths = {}
        file_root_paths = {}

        for path in paths:
            file_name = path.parts[-1]
            category_id = re.match(self.category_pattern, file_name).group(0)
            file_id = re.match(self.file_pattern, file_name).group(0)

            file_root_paths[file_id] = (category_id, file_id, path)
            d = {file_id: (category_id, file_id, path)}

            if category_id in cat_root_paths:
                cat_root_paths[category_id].update(d)
            else:
                cat_root_paths[category_id] = d

        self.cat_root_paths = cat_root_paths
        self.file_root_paths = file_root_paths

    def _load_objects(self, category_id=None, file_id=None):
        """
        Load text from paths or tuples and feed via generator

        Parameters
        ----------
        category_id : str
            category id chosen to stream from
        file_id : str
            file_id chosen to stream from

        ----------
        Returns
            generator
                category_id,file_id and _block from paths or tuples

        """

        if category_id:
            load_paths = list(self.cat_root_paths[category_id].values())
        elif file_id:
            load_paths = [self.file_root_paths[file_id]]
        else:
            load_paths = list(self.file_root_paths.values())

        if self.isroot:
            for _catgeory_id, _file_id, _values in load_paths:
                with open(_values, "r", encoding=self.encoding) as text:
                    while text.tell() < os.stat(_values).st_size:
                        _block = self.read_block(text)
                        yield (_catgeory_id, _file_id, _block)
        else:

            for _catgeory_id, _file_id, _values in load_paths:
                _block = self.read_block(io.StringIO(_values))
                yield (_catgeory_id, _file_id, _block)

    def _stem(self, tokens):
        """take in pre-tokenized tokens"""
        if isinstance(tokens, str):
            tokens = [tokens]

        stem_tokens = []
        for token in tokens:

            if token.lower() not in self.stop_words and not token.isdigit():
                stem_tokens.append(self.stemmer.stem(token.lower()))

        if stem_tokens:
            if len(stem_tokens) == 1:
                return stem_tokens[0]
            else:
                return stem_tokens

    def _lemmatize(self, tokens):
        """take in pre-tokenized tokens"""
        if isinstance(tokens, str):
            tokens = [tokens]

        lem_tokens = []
        token_tag_pairs = pos_tag(tokens)
        for t in token_tag_pairs:
            token, tag = t
            if token.lower() not in self.stop_words and not token.isdigit():
                lem_tokens.append(
                    self.lemmatizer.lemmatize(token.lower(), pos=custom_get_pos(tag))
                )

        if lem_tokens:
            if len(lem_tokens) == 1:
                return lem_tokens[0]
            else:
                return lem_tokens

    def get_words(self, file_id=None, category_id=None, stem=False, lemmatize=False):
        """
        Take in block of text and return tokenized words

        Parameters
        ----------
        category_id : str
            category id chosen to stream from
        file_id : str
            file_id chosen to stream from
        stem : boolean
            store stemmed versions of tokens
        lemmatize : boolean
            store lemmatized versions of tokens



        Returns
        ----------
        Corpus instance


        """

        if stem and not self.stemmer:
            raise ValueError("Missing stemmer")
        if lemmatize and not self.lemmatizer:
            raise ValueError("Missing lemmatizer")

        base_tokenizer = self.word_tokenizer.tokenize

        words = []
        structure = "word"

        for _category_id, _file_id, _block in self._load_objects(
            category_id=category_id, file_id=file_id
        ):

            block_tokenized = base_tokenizer(_block)

            for tokens in block_tokenized:
                block_tokens = []
                stems = []
                lems = []

                if tokens:
                    block_tokens.append(tokens)

                    if stem:
                        stemmed_tokens = self._stem(tokens)
                        if stemmed_tokens:
                            stems.append(stemmed_tokens)

                    if lemmatize:
                        lemmed_tokens = self._lemmatize(tokens)
                        if lemmed_tokens:
                            lems.append(lemmed_tokens)

                document = self.Document(
                    category_id=_category_id,
                    file_id=_file_id,
                    raw=block_tokens,
                    tokens=tokens,
                    stem=stems,
                    lemma=lems,
                    structure=structure,
                )

                words.append(document)

        return self.Corpus(words)

    def get_sents(self, file_id=None, category_id=None, stem=False, lemmatize=False):
        """
        Take in block of text and return tokenized sentence-words

        Parameters
        ----------
        category_id : str
            category id chosen to stream from
        file_id : str
            file_id chosen to stream from
        stem : boolean
            store stemmed versions of tokens
        lemmatize : boolean
            store lemmatized versions of tokens



        Returns
        ----------
        Corpus instance


        """

        if stem and not self.stemmer:
            raise ValueError("Missing stemmer")
        if lemmatize and not self.lemmatizer:
            raise ValueError("Missing lemmatizer")

        base_tokenizer = self.sent_tokenizer.tokenize
        secondary_tokenizer = self.word_tokenizer.tokenize

        sents = []
        structure = "sentence"

        for _category_id, _file_id, _block in self._load_objects(
            category_id=category_id, file_id=file_id
        ):

            block_tokenized = base_tokenizer(_block)

            for tokens in block_tokenized:
                raw_block = tokens
                block_tokens = []
                stems = []
                lems = []

                tokens = secondary_tokenizer(tokens)
                if tokens:
                    block_tokens.extend(tokens)

                    if stem:
                        stemmed_tokens = self._stem(tokens)
                        if stemmed_tokens:
                            stems.extend(stemmed_tokens)

                    if lemmatize:
                        lemmed_tokens = self._lemmatize(tokens)
                        if lemmed_tokens:
                            lems.extend(lemmed_tokens)

                    document = self.Document(
                        category_id=_category_id,
                        file_id=_file_id,
                        raw=raw_block,
                        tokens=block_tokens,
                        stem=stems,
                        lemma=lems,
                        structure=structure,
                    )

                    sents.append(document)

        return self.Corpus(sents)

    def get_paras(
        self, file_id=None, category_id=None, flatten=False, stem=False, lemmatize=False
    ):
        """
        Take in block of text and return tokenized sentence-words

        Parameters
        ----------
        category_id : str
            category id chosen to stream from
        file_id : str
            file_id chosen to stream from
        flatten : boolean
            if True keep documents at 1 dimenion
            e.g.
            when False : [['This' ,'is' 'a','tokenized','sentence'],['and','another]]
            when True : ['This' ,'is' 'a','tokenized','sentence''and','another]

        stem : boolean
            store stemmed versions of tokens
        lemmatize : boolean
            store lemmatized versions of tokens


        Returns
        ----------
        Corpus instance


        """

        if stem and not self.stemmer:
            raise ValueError("Missing stemmer")
        if lemmatize and not self.lemmatizer:
            raise ValueError("Missing lemmatizer")

        paras = []

        base_tokenizer = self.sent_tokenizer.tokenize
        secondary_tokenizer = self.word_tokenizer.tokenize

        if flatten:
            base_tokenizer = self.word_tokenizer.tokenize
            secondary_tokenizer = None

        structure = "paragraph"

        for _category_id, _file_id, _block in self._load_objects(
            category_id=category_id, file_id=file_id
        ):
            raw_block = _block
            block_tokenized = base_tokenizer(_block)
            block_tokens = []
            stems = []
            lems = []

            for tokens in block_tokenized:
                if secondary_tokenizer:
                    tokens = secondary_tokenizer(tokens)

                if tokens:
                    block_tokens.append(tokens)

                    if stem:
                        stemmed_tokens = self._stem(tokens)
                        if stemmed_tokens:
                            stems.append(stemmed_tokens)

                    if lemmatize:
                        lemmed_tokens = self._lemmatize(tokens)
                        if lemmed_tokens:
                            lems.append(lemmed_tokens)

            document = self.Document(
                category_id=_category_id,
                file_id=_file_id,
                raw=raw_block,
                tokens=block_tokens,
                stem=stems,
                lemma=lems,
                structure=structure,
            )

            paras.append(document)

        return self.Corpus(paras)

    def truncate_text(
        self,
        start_regex,
        end_regex,
        keep_start_end=False,
        overwrite=False,
        folder_prefix="_truncated",
        return_stats=False,
    ):
        """
        Load in texts and truncate. Save in a different
        location depending on parameters

        Parameters
        ----------
        start_regex : str|re.compile object
            regex to start crop from
        end_regex : str |re.compile object
            regex to end truncation from
        keep_start_end : boolean
            if True, only consider truncation from end of match

        overwrite : boolean
            choose to overwrite current directory
        folder_prefix : boolean
            when overwrite = False, create a new folder with prefix
        return_stats : boolean
            return

        Returns
        ----------
        file_stats : dict
            when file_stats is true



        """

        self.root = self.init_root
        root_child = self.root.parts[-1]
        if not overwrite:
            new_root = self.root.parent.joinpath(root_child + folder_prefix)
        else:
            new_root = self.root

        if os.path.exists(new_root):
            print("Overwritting existing folder")
        else:
            os.mkdir(new_root)
            print(r"Intializing new path ./{}".format(root_child + folder_prefix))

        if keep_start_end:
            start_slice, end_slice = 0, -1
        else:
            start_slice, end_slice = -1, 0

        file_stats = {}
        for _, _, p in self.file_root_paths.values():
            file_name = p.parts[-1]
            with open(p, "r", encoding=self.encoding) as f:
                p_read = f.read()
                org_size = len(p_read)

                start_pos = re.search(start_regex, p_read)
                if start_pos:
                    start_pos = start_pos.span()[start_slice]
                    p_read = p_read[start_pos:]
                else:
                    print(
                        "start_regex could not be found in {}, defaulting to start of file".format(
                            p.parts[-1]
                        )
                    )

                end_pos = re.search(end_regex, p_read)
                if end_pos:
                    end_pos = end_pos.span()[end_slice]
                    p_read = p_read[0:end_pos]
                else:
                    print(
                        "end_regex could not be found in {}, defaulting to end of file".format(
                            p.parts[-1]
                        )
                    )
                f.close()

            with open(new_root.joinpath(file_name), "w", encoding=self.encoding) as f:
                f.write(p_read)
                f.close()
            file_stats[file_name] = 100 * (len(p_read) / org_size - 1)

        self.root = new_root
        self._read_paths()

        if return_stats:
            return file_stats

    @staticmethod
    def read_block(text):
        block = ""
        while True:
            line = text.readline()
            if not line:
                if block:
                    return block
                else:
                    return ""
            elif line and not line.strip():
                if block:
                    return block
            else:
                block += line
