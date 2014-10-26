from wapiti import Model
from nltk.tag import TaggerI

class WapitiPOSTagger(TaggerI):
	"""docstring for WapitiPOSTagger"""
	def __init__(self, *args, **kwargs):
		if 'model' not in kwargs:
			kwargs['model'] = 'resources/model.txt'
		if 'pattern' not in kwargs:
			kwargs['pattern'] = 'resources/pattern.txt'
		super(WapitiPOSTagger, self).__init__()

		option_dict = {}
		option_dict['pattern'] = kwargs['pattern']
		option_dict['model'] = kwargs['model']
		self.model = Model(**option_dict)

	def tag_sents(self, sents):
		for words in sents:
			tags = self.model.label_sequence('\n'.join(words)).split('\n')
			yield zip(words, tags)

	def tag(self, sent):
		tags = self.model.label_sequence('\n'.join(sent)).split('\n')
		return zip(sent, tags)