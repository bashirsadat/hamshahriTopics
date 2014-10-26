import multiprocessing
import time
import sys

from pymongo import MongoClient
from hazm import sent_tokenize, word_tokenize, POSTagger
from settings import Settings


def load_stopwords():
	stopwords = {}
	with open('resources/PersianStopwords.txt', 'rU') as f:
		for line in f:
			stopwords[line.strip()] = 1

	return stopwords


def worker(identifier, skip, count):
	tagger = POSTagger()
	done = 0
	start = time.time()
	stopwords = load_stopwords()
	documents_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.HAMSHAHRI_DATABASE][
		Settings.HAMSHAHRI_COLLECTION]
	tags_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.TAGS_DATABASE][
		Settings.HAMSHAHRI_COLLECTION]

	batch_size = 50
	for batch in range(0, count, batch_size):
		hamshahri_cursor = documents_collection.find().skip(skip + batch).limit(batch_size)
		for doc in hamshahri_cursor:
			words = []
			sentences = sent_tokenize(doc['text'])
			sents = []
			for sentence in sentences:
				tokens = word_tokenize(sentence)
				text = [word for word in tokens if word not in stopwords]
				sents.append(text)

			tags = tagger.tag_sents(sents)
			for sent in tags:
				for word, tag in sent:
					words.append({'word': word, "pos": tag})

			tags_collection.insert({
				"id": doc["id"],
				"categories_fa": doc["categories_fa"],
				"text": doc["text"],
				"words": words
			})

			done += 1
			#if done % 100 == 0:
			end = time.time()
			print 'Worker' + str(identifier) + ': Done ' + str(done) + ' out of ' + str(count) + ' in ' + (
				"%.2f" % (end - start)) + ' sec ~ ' + ("%.2f" % (done / (end - start))) + '/sec'
			sys.stdout.flush()


def main():
	hamshahri_collection = MongoClient(Settings.MONGO_CONNECTION_STRING)[Settings.HAMSHAHRI_DATABASE][
		Settings.HAMSHAHRI_COLLECTION]
	hamshahri_cursor = hamshahri_collection.find()
	count = hamshahri_cursor.count()
	workers = 1
	batch = count / workers

	jobs = []
	for i in range(workers):
		p = multiprocessing.Process(target=worker, args=((i + 1), i * batch, count / workers))
		jobs.append(p)
		p.start()

	for j in jobs:
		j.join()
		print '%s.exitcode = %s' % (j.name, j.exitcode)


if __name__ == '__main__':
	main()
	12180