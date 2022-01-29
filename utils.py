import re
import os
from pathlib import Path
from nltk import pos_tag as pos_tag
import io
import json
import torch
from nltk.corpus import wordnet
import numpy as np


def custom_get_pos(first_tag):

    if first_tag.startswith('J'):
        return wordnet.ADJ
    elif first_tag.startswith('V'):
        return wordnet.VERB
    elif first_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class Document(object):
    ''' A document can be a paragraph, sentence or raw text
        Each document comes from splitting original corpus 
        This will either be tokenized or raw, not nested lists'''
    #We want to be able to loop through each document, and extra attributes
    def __init__(self,category_id,file_id,text,lemma,stem,unique_id=None,structure='sentence'):
        self.category_id = category_id
        self.file_id = file_id
        self.unique_id = unique_id
        self.text = text
        self.stem = stem
        self.lemma = lemma
        self.structure = structure


    def __str__(self):
        return str(self.text)
    def __repr__(self):
        return str(self.text)
    def __getitem__(self,index):
        ''' default to text'''
        return self.text[index]
    def __len__(self):
        return len(self.text)
    def to_json(self):
        return json.dumps(self,default=lambda o:o.__dict__,sort_keys=True,indent=4)




class CorpusPreProcess(object):
    ''' CorpusPreProcessV2 either takes in a root or a list of text with ids
        This also needs to allow for categories'''
    def __init__(self,root,file_extension,
                        category_pattern,
                        file_pattern,
                        word_tokenizer,
                        sent_tokenizer,
                        lemmatizer=None,
                        stemmer=None,
                        block_reader=None,
                        stop_words=[],
                        encoding='utf-8'):

        self.file_extension = file_extension
        self.file_pattern = file_pattern
        self.category_pattern = category_pattern
        self.word_tokenizer = word_tokenizer
        self.sent_tokenizer = sent_tokenizer
        self.encoding=encoding
        self.lemmatizer = lemmatizer
        self.stemmer = stemmer
        self.stop_words = stop_words
        self.file_ids = None
        self.Document = Document
        

        if block_reader:
            self.block_reader = block_reader

        self.root_paths = None
        
        if isinstance(root,str):
            if os.path.exists(root):
                self.root = Path(root)#root is a path
                #create original root for truncation
                self.init_root = self.root
                self.isroot = True
        elif isinstance(root,list):
            if not all(isinstance(i,tuple)*(len(i)==3) for i in root):
                raise ValueError("Input text needs nested tuples [(id,document)]")
            else:
                self.isroot=False
                self.root = root
        else:
            raise ValueError("root needs an existing address or a nested list of tuples")

        if self.isroot:
            self._read_paths()
        else:
            #begin loading in text
            self._read_tuples()

        self._file_ids = list(self.file_root_paths.keys())
        self._category_ids = list(self.cat_root_paths.keys())

    def get_file_ids(self,category_id=None):
        if category_id:
            return list(self.cat_root_paths[category_id].keys())
        else:
            return list(self.file_root_paths.keys())

    def get_category_ids(self):
        return self._category_ids

    def _read_tuples(self):
        cat_root_paths = {}
        file_root_paths = {}

        for category_id,file_id,item in self.root:
            file_root_paths[file_id] = (category_id,file_id,item)
            d = {file_id:(category_id,file_id,item)}

            if category_id in cat_root_paths:
                cat_root_paths[category_id].update(d)
            else:
                cat_root_paths[category_id] = d

        self.cat_root_paths = cat_root_paths
        self.file_root_paths = file_root_paths

    def _read_paths(self):

        paths = sorted(self.root.glob(self.file_extension))

        cat_root_paths = {}
        file_root_paths = {}

        for path in paths:
            file_name = path.parts[-1]
            category_id = re.match(self.category_pattern,file_name).group(0)
            file_id = re.match(self.file_pattern,file_name).group(0)

            file_root_paths[file_id]=(category_id,file_id,path)
            d = {file_id:(category_id,file_id,path)}
            
            if category_id in cat_root_paths:
                cat_root_paths[category_id].update(d)
            else:
                cat_root_paths[category_id] = d

        self.cat_root_paths = cat_root_paths
        self.file_root_paths = file_root_paths
 
            
    def _load_objects(self,category_id=None,file_id=None):
 
        if category_id:
            load_paths = list(self.cat_root_paths[category_id].values()) #This outputs a set of dictionaries
        elif file_id:
            load_paths = [self.file_root_paths[file_id]]
        else:
            load_paths = list(self.file_root_paths.values())

        blocks = []
        if self.isroot:
            for _catgeory_id,_file_id,_values in load_paths:
                with open(_values,'r',encoding=self.encoding) as text:
                    while text.tell()<os.stat(_values).st_size:
                        _block = self.read_block(text)
                        yield (_catgeory_id,_file_id,_block)
        else:

            for _catgeory_id,_file_id,_values in load_paths:
                _block = self.read_block(io.StringIO(_values))
                yield (_catgeory_id,_file_id,_block)

      
    def _stem(self,tokens): # what if not flat?
        '''take in pre-tokenized tokens '''
        if isinstance(tokens,str):
            tokens = [tokens]

        stem_tokens = []
        for token in tokens:

            if token.lower() not in self.stop_words and not token.isdigit():
                #return self.stemmer.stem(token.lower())
                stem_tokens.append(self.stemmer.stem(token.lower()))

        if stem_tokens:
            if len(stem_tokens)==1:
                return stem_tokens[0]
            else:
                return stem_tokens
        


    def _lemmatize(self,tokens):
        ''' take in pre-tokenized tokens'''
        if isinstance(tokens,str):
            tokens = [tokens]

        lem_tokens = []
        token_tag_pairs = pos_tag(tokens)
        for t in token_tag_pairs:
            token,tag = t
            if token.lower() not in self.stop_words and not token.isdigit():
                lem_tokens.append(self.lemmatizer.lemmatize(token.lower(),pos=custom_get_pos(tag)))
        
        if lem_tokens:
            if len(lem_tokens)==1:
                return lem_tokens[0]
            else:
                return lem_tokens


    def get_words(self,file_id=None,category_id=None,stem=False,lemmatize=False):

        
        if stem and not self.stemmer:
            raise ValueError("Missing stemmer")
        if lemmatize and not self.lemmatizer:
            raise ValueError("Missing lemmatizer")

        base_tokenizer = self.word_tokenizer.tokenize
    

        words = []
        structure = 'word'
 
        for _category_id,_file_id,_block in self._load_objects(category_id=category_id,file_id=file_id):
            #for each path
            block_tokenized = base_tokenizer(_block) # returns list of text, we tokenize either to words ['a','b','c'] or sentences ['aaa','bbbb bbb','cccc ccc ']
            
            
            for tokens in block_tokenized: # loop through each token or sentence
                text = []
                stems = []
                lems = []

                if tokens:
                    text.append(tokens)
            
                    if stem: #keep if statements in loop for readability/avoid nested lambda, if tokens then 
                        stemmed_tokens = self._stem(tokens) 
                        if stemmed_tokens:
                            stems.append(stemmed_tokens)
        
                    if lemmatize:
                        lemmed_tokens = self._lemmatize(tokens)
                        if lemmed_tokens:
                            lems.append(lemmed_tokens)

            
                document = Document(category_id=_category_id,
                                    file_id =_file_id,
                                    unique_id=None,
                                    text=text,
                                    stem = stems,
                                    lemma=lems,
                                    structure=structure)

                words.append(document)
                        
        return words

    def get_sents(self,file_id=None,category_id=None,stem=False,lemmatize=False):

        
        if stem and not self.stemmer:
            raise ValueError("Missing stemmer")
        if lemmatize and not self.lemmatizer:
            raise ValueError("Missing lemmatizer")

        base_tokenizer = self.sent_tokenizer.tokenize
        secondary_tokenizer =  self.word_tokenizer.tokenize


        sents = []
        structure = 'sentence'
 
        for _category_id,_file_id,_block in self._load_objects(category_id=category_id,file_id=file_id):
            
            block_tokenized = base_tokenizer(_block) 
            
            for tokens in block_tokenized: 
                text = []
                stems = []
                lems = []       
                tokens = secondary_tokenizer(tokens)
                if tokens:
                    text.extend(tokens)
                    if stem: #keep if statements in loop for readability/avoid nested lambda, if tokens then 
                            stemmed_tokens = self._stem(tokens) 
                            if stemmed_tokens:
                                stems.extend(stemmed_tokens)
            
                    if lemmatize:
                        lemmed_tokens = self._lemmatize(tokens)
                        if lemmed_tokens:
                            lems.extend(lemmed_tokens)

                    document = Document(category_id=_category_id,
                                    file_id =_file_id,
                                    unique_id=None,
                                    text=text,
                                    stem = stems,
                                    lemma=lems,
                                    structure=structure)

                    sents.append(document)
                
        return sents

    
    def get_paras(self,file_id=None,category_id=None,flatten=False,stem=False,lemmatize=False):

        
        if stem and not self.stemmer:
            raise ValueError("Missing stemmer")
        if lemmatize and not self.lemmatizer:
            raise ValueError("Missing lemmatizer")

        base_tokenizer = self.sent_tokenizer.tokenize
        secondary_tokenizer =  self.word_tokenizer.tokenize

        if flatten:
            base_tokenizer = self.word_tokenizer.tokenize
            secondary_tokenizer = None

        paras = []
        structure = 'paragraph'
 
        for _category_id,_file_id,_block in self._load_objects(category_id=category_id,file_id=file_id):
            #for each path
            block_tokenized = base_tokenizer(_block) # returns list of text, we tokenize either to words ['a','b','c'] or sentences ['aaa','bbbb bbb','cccc ccc ']
            text = []
            stems = []
            lems = []
            
            for tokens in block_tokenized: # loop through each token or sentence


                if secondary_tokenizer: # if secondary tokenizer is used, then  our tokens our tokenized sentences
                    tokens = secondary_tokenizer(tokens)

                if tokens:
                    text.append(tokens)
            
                    if stem: #keep if statements in loop for readability/avoid nested lambda, if tokens then 
                        stemmed_tokens = self._stem(tokens) 
                        if stemmed_tokens:
                            stems.append(stemmed_tokens)
        
                    if lemmatize:
                        lemmed_tokens = self._lemmatize(tokens)
                        if lemmed_tokens:
                            lems.append(lemmed_tokens)

            
            document = Document(category_id=_category_id,
                                file_id =_file_id,
                                unique_id=None,
                                text=text,
                                stem = stems,
                                lemma=lems,
                                structure=structure)

            paras.append(document)
                    
        return paras
    


    def truncate_text(self,regex,overwrite=False,folder_prefix='_truncated'):
        '''truncate text using regex, output into new folder or overwrite'''
        #reset root to initialized root
        self.root = self.init_root
        root_child = self.root.parts[-1]
        if overwrite==False:
            new_root = self.root.parent.joinpath(root_child+folder_prefix)
        else:
            new_root = self.root
            
        if os.path.exists(new_root):
            print("Overwritting existing folder")
        else:
            os.mkdir(new_root)

        for _,_,p in self.file_root_paths.values():
            file_name = p.parts[-1]
            with open(p,'r',encoding=self.encoding) as f:
                p_read = f.read()
                try:
                    start,end = re.search(regex,p_read).span()
                    p_read = p_read[start:end]
                    f.close()
                except Exception as e:
                    print("regex could not truncate document {}".format(p.parts[-1]))
                    f.close()
                
            
            with open(new_root.joinpath(file_name),'w') as f:
                f.write(p_read)
                f.close()

        
        self.root = new_root
        self._read_paths()


    @staticmethod
    def read_block(text):
        block = ""
        while True:
            line = text.readline()
            if not line:
                if block:
                    return block
                else:
                    return ''
            elif line and not line.strip():
                if block:
                    return block
            else:
                block += line


class TopicSimSen(object):
    ''' 
    The purpose of this class is to allow for the following things:
    input -> [str], str,CorpusDocumentObject,Path
    output -> Depending on method,
    1)What is my sentiment? 
    2)What is my topic?
    3)What is my similarity across other embeddings

    '''
    sent_dict = {0: 'positive', 1: 'negative', 2: 'neutral'}

    def __init__(self,corpus,
                    transformer_model,
                    transformer_tokenizer,
                    tokenizer_settings,
                    lda_model,
                    lda_vec,
                    lda_topics,
                    batch_size=20,
                    corpus_pre_process_compare=None,structure=''):


        self.corpus = corpus 
        self.transformer_model = transformer_model
        self.transformer_tokenizer = transformer_tokenizer
        self.tokenizer_settings = tokenizer_settings
        self.lda_model = lda_model
        self.lda_vec = lda_vec
        self.lda_topics = lda_topics
        self.batch_size=batch_size


        self.transformer_class = self.transformer_model.__class__

    def _get_topic_output(self):
        for n,batch in enumerate((self.corpus[i:i+self.batch_size] for i in range(0,len(self.corpus),self.batch_size))):
            bag_of_words = self.lda_vec.transform(batch)
            lda_topics = self.lda_model.transform(bag_of_words)
            yield lda_topics


    def _get_model_output(self):

        for n,batch in enumerate((self.corpus[i:i+self.batch_size] for i in range(0,len(self.corpus),self.batch_size))):
             #avoid error but we need these objects
            feature_tensor = self.transformer_tokenizer([b.text for b in batch],**self.tokenizer_settings)
            with torch.no_grad():
                    output = self.transformer_model(**feature_tensor)
                    yield (feature_tensor,output) # return every batch of results

    def _get_sentiment(self,model_output):
        for _,_model_output in model_output:

            logits = self.softmax(np.array(_model_output[0]))
            if logits.shape[0] == 1:
                predictions = np.argmax(logits, axis=1)
            else:
                predictions = np.squeeze(np.argmax(logits, axis=1))

            yield (logits,predictions)

    def _get_doc_embedding(self,model_output):
        for _feature_tensor,_model_output in model_output:

            embedding_from_last = _model_output.hidden_states[-1]
            attention_mask = _feature_tensor['attention_mask']
            doc_embedding = self._embedding_max_pool(embedding_from_last,attention_mask)

            yield doc_embedding

    def get_sentiment(self,return_corpus=True):
        
  
        corpus = []
        batch_start = 0
        for _logits,_predictions in self._get_sentiment(self._get_model_output()):
            batch_n = len(_logits)
            corpus_batch = self.corpus[batch_start:batch_start+batch_n]

            for n,doc in enumerate(corpus_batch):
                logits = _logits[n].tolist()
                predictions = _predictions[n]
                doc = corpus_batch[n]
                doc.sentiment = {'logits':logits,'predictions':TopicSimSen.sent_dict[predictions]}
                corpus.append(doc)

            batch_start+= batch_n

        if return_corpus:

            return corpus

    def get_doc_embedding(self,return_corpus=True):
        

        corpus = []
        batch_start = 0
        for _doc_embedding in self._get_doc_embedding(self._get_model_output()):
            batch_n = len(_doc_embedding)
            corpus_batch = self.corpus[batch_start:batch_start+batch_n]

            for n,doc in enumerate(corpus_batch):
                doc_embedding = _doc_embedding[n].squeeze().tolist() #toseralize
                doc = corpus_batch[n]
                doc.embedding = {'embedding':doc_embedding,'model_class':self.transformer_class}
            
                corpus.append(doc)

            batch_start+= batch_n

        if return_corpus:

            return corpus

    def get_topics(self,return_corpus=True):
        corpus = []
        batch_start = 0
        for topic_dist in self._get_topic_output():
            batch_n = len(topic_dist)
            corpus_batch = self.corpus[batch_start:batch_start+batch_n]

            for n,doc in enumerate(corpus_batch):
                topic = topic_dist[n]
                max_topic = self.lda_topics[topic.argmax()]
                doc = corpus_batch[n]
                doc.topics = {'topic_dist':topic.tolist(),'topic_max':max_topic}
                corpus.append(doc)

            batch_start+= batch_n

        return corpus

    def get_features(self,sentiment=True,embedding=True,topic=True):
        """
        get_features returns corpus with sentiment,embedding or topic attributes

        sentiment: extract sentiment
        embedding: extract transformer word embeddings
        topic: extract topics

        This function can be made efficient by shareing
        model generator between get_sentiment and 
        get_doc_embedding methods

        """

        if sentiment:
            self.get_sentiment(return_corpus=False)
        
        if embedding:
            self.get_doc_embedding(return_corpus=False)
        
        if topic:
            self.get_topics(return_corpus=False)
        
        return self.corpus

    @staticmethod
    def _embedding_max_pool(embedding_from_last,attention_mask):
        """
        _embedding_max_pool 
        
        
        """
        mask = attention_mask.unsqueeze(-1).expand(embedding_from_last.shape).float()
        mask_embeddings =  embedding_from_last * mask

        return torch.sum(mask_embeddings,1)/torch.clamp(mask.sum(1),min=1e-8)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / np.sum(e_x, axis=1)[:, None]


