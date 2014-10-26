#encoding=utf8
from __future__ import unicode_literals
from gensim import corpora, models
from operator import itemgetter
from hazm import HamshahriReader, sent_tokenize, word_tokenize, POSTagger
import codecs, joblib, time


class HamshahriTopicModels:
	def __init__(self, path='resources/hamshahri'):
		self.hamshahri = HamshahriReader('resources/hamshahri')
		self.stopwords = {}
		with codecs.open('resources/PersianStopwords.txt', 'rU', encoding='utf8') as f:
			for line in f:
				self.stopwords[line.strip()] = 1

	def texts(self, categories={'Politics'}, limit=None):
		docs = self.hamshahri.docs()
		print 'start reading corpus...'
		count = 0
		texts = []
		for doc in docs:
			if limit is not None and count == limit:
				break
			if len(categories.intersection(set(doc["categories_en"]))) > 0:
				count += 1
				for sent in sent_tokenize(doc['text']):
					if len(sent) <= 1:
						continue
					texts.append([word for word in word_tokenize(sent) if word not in self.stopwords and len(word) > 1])
		return texts

	def nouns(self, texts):
		total_count = len(texts)
		tagger = POSTagger()
		nouns = []
		tagged_doc = tagger.tag_sents(texts)
		for sent in tagged_doc:
			sentence = []
			for word, tag in sent:
				if tag == 'N':
					sentence.append(word)
			nouns.append(sentence)

		return nouns



	def ldaModel(self, texts, n_topics=30):
		self.dictionary = corpora.Dictionary(texts)
		corpus = [self.dictionary.doc2bow(text) for text in texts]
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		return models.LdaModel(corpus_tfidf, id2word=self.dictionary, num_topics=n_topics), self.dictionary

	def lsiModel(self, texts, n_topics=30):
		dictionary = corpora.Dictionary(texts)
		corpus = [dictionary.doc2bow(text) for text in texts]
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		return models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics), self.dictionary


	def printTopics(self, model, n_topics=30, n_terms=10):
		result = []
		for i in range(0, n_topics):
			temp = model.show_topic(i, n_terms)
			result.append(temp)
			terms = []
			for term in temp:
				terms.append(term[1])
			print "Top 10 terms for topic #" + str(i) + ": " + ", ".join(terms)
		return result

	def topics(self, model, document, dictionary=None):
		if dictionary is not None:
			self.dictionary = dictionary
		text = [w for w in word_tokenize(document) if w not in self.stopwords and len(w) > 1]
		corpus = self.dictionary.doc2bow(text)
		print 'Which LDA topic maximally describes a document?\n'
		print 'Original document: ' + document
		print 'Topic probability mixture: ' + str(model[corpus])
		print 'Maximally probable topic: topic #' + str(max(model[corpus], key=itemgetter(1))[0])
		return model[corpus]

"""
hamshahriTopics = HamshahriTopicModels()
#texts = joblib.load('hamshahriPolitics.txt')
print 'texts loaded...'
dictionary = joblib.load('hamshahriPolitics.dictionary')
print 'dictionary loaded...'
model = joblib.load('LDAPolitics.model')
print 'model loaded...'
test = codecs.open('tests/test04.txt', 'r', encoding='utf8').read()
hamshahriTopics.topics(model, test, dictionary)
"""

hamshahriTopics = HamshahriTopicModels()
texts = joblib.load('hamshahriPolitics.txt')
print 'texts loaded...'
hamshahriPoliticsNouns = hamshahriTopics.nouns(texts)
print 'Nouns generated...'
joblib.dump(hamshahriPoliticsNouns, 'hamshahriPoliticsNouns.txt')
print 'nouns saved successfully'



