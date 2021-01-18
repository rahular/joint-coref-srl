import os
import nltk
import ast
import logging

import numpy as np

from collections import Counter, defaultdict, OrderedDict

logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)
logger = logging.getLogger(__name__)

# doc. length wise errors over all datasets
len_errors = {
	"0-100": list(),
	"100-200": list(),
	"200-300": list(),
	"300-400": list(),
	"400-500": list(),
	"500-600": list(),
	"600-700": list(),
	"700-800": list(),
	"800-900": list(),
	"900-1000": list(),
	"1000+": list(),
}

def readFile(path):
	"""
	path: string path to file for a given domain
	"""
	shitty = 0
	total = 0
	list_of_document_dicts = []
	with open(path, "r") as infile:
		txt = infile.read()
		documents = txt.split(os.linesep + os.linesep)
		for i, doc in enumerate(documents):
			try:
				doc = doc.split(os.linesep)
				tokenized_doc = doc[1]
				missing_mentions = doc[2]
				predictions = doc[4:]
				predictions = [i.split("\t") for i in predictions]
				checked_predictions = [p for p in predictions if len(p) == 3]
				for p in predictions:
					total += 1
					if len(p) != 3:
						logging.debug(p)
						shitty += 1
				docdict = {
					"tokenized_doc": ast.literal_eval(tokenized_doc),
					"missing": missing_mentions,
					"predictions": checked_predictions,
				}
				list_of_document_dicts.append(docdict)
			except:
				logging.debug(
					f"* problematic doc #{i}"
				)  # this should only be a blank document at the end of the file
				logging.debug(doc)
				logging.debug("#####" * 10)
				pass
	logging.debug(f"* Bad predictions skipped: {shitty} / {total}")
	return list_of_document_dicts


def countErrorsCoref(list_of_document_dicts):
	total_predictions = 0
	total_correct = 0
	total_errors = 0

	pronoun_errors = 0  # PRP, PRP$
	noun_error = 0  # NN, NNS
	pnoun_error = 0  # NNP, NNPS
	verb_error = 0  # VB, VBD, VBG, VBN, VBP, VBZ
	wh_error = 0  # WDT, WRB
	poss_wh_error = 0  # WP, WP$
	mwe_error = 0  # MWE
	other_error = 0
	other_error_tags = []  # hmmm what are these
	bad_pred_len_dict = defaultdict(int)

	for d in list_of_document_dicts:
		doc_errors, doc_correct = 0, 0
		word_tags = nltk.pos_tag([word for word, *_ in d["predictions"]])
		for (word, gold, pred), (_, tag) in zip(d["predictions"], word_tags):
			gold = gold.strip()
			pred = pred.strip()
			try:
				if " " in word:
					tag = "MWE"  # multi word expression
				total_predictions += 1

				if pred in gold:
					doc_correct += 1
				else:
					doc_errors += 1
					bad_pred_len_dict[str(len(word))] += 1

					if tag in ["PRP", "PRP$"]:
						pronoun_errors += 1
					elif tag in ["NN", "NNS"]:
						noun_error += 1
					elif tag in ["NNP", "NNPS"]:
						pnoun_error += 1
					elif tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
						verb_error += 1
					elif tag in ["WDT", "WRB"]:
						wh_error += 1
					elif tag in ["WP", "WP$"]:
						poss_wh_error += 1
					elif tag in ["MWE"]:
						mwe_error += 1
					else:
						other_error += 1
						other_error_tags.append(tag)
			except Exception:
				logging.debug(f"* stupid problem word {word}")
		total_errors += doc_errors
		total_correct += doc_correct

		def get_bucket_key(l):
			if l < 100: return "0-100"
			if l >= 100 and l < 200: return "100-200"
			if l >= 200 and l < 300: return "200-300"
			if l >= 300 and l < 400: return "300-400"
			if l >= 400 and l < 500: return "400-500"
			if l >= 500 and l < 600: return "500-600"
			if l >= 600 and l < 700: return "600-700"
			if l >= 700 and l < 800: return "700-800"
			if l >= 800 and l < 900: return "800-900"
			if l >= 900 and l < 1000: return "900-1000"
			else: return "1000+"
		
		bucket_key = get_bucket_key(len(d["tokenized_doc"]))
		if len(d["predictions"]) >= 5:
			len_errors[bucket_key].append(100 * (doc_correct / len(d["predictions"])))

	assert total_correct + total_errors == total_predictions

	# The order is: ["pronouns", "nouns", "proper-nouns", "verbs", "wh", "possessive-wh", "mwe", "other"]
	pos_errors = [
		pronoun_errors,
		noun_error,
		pnoun_error,
		verb_error,
		wh_error,
		poss_wh_error,
		mwe_error,
		other_error,
	]
	all_errors = total_errors / total_predictions

	bad_pred_len_dict = OrderedDict(sorted(bad_pred_len_dict.items(), key=lambda t: int(t[0])))

	return all_errors, pos_errors, bad_pred_len_dict


def pretty_print(output):
	keys = sorted(list(output.keys()))
	logging.info("All errors")
	logging.info(keys)
	logging.info(["{:.2f}".format(output[key]["all_errors"] * 100) for key in keys])
	logging.info("POS errors")
	logging.info([output[key]["pos_errors"] for key in keys])
	logging.info([(key, output[key]["span_len_errors"]) for key in keys])
	le = {k: np.array(v) for k, v in len_errors.items()}
	le = {k: ("{:.2f}".format(np.mean(v)), "{:.2f}".format(np.std(v))) for k, v in le.items()}
	logging.info(le)


def main(dir):
	output = dict()
	coref_files = [
		f
		for f in os.listdir(dir)
		if os.path.isfile(os.path.join(dir, f)) and f.startswith("coref")
	]
	for f in coref_files:
		dataset = "_".join(f.split(".")[0].split("_")[4:])
		list_docs = readFile(os.path.join(dir, f))
		all_errors, pos_errors, span_len_errors = countErrorsCoref(list_docs)
		output[dataset] = {"all_errors": all_errors, "pos_errors": pos_errors, "span_len_errors": span_len_errors}
	pretty_print(output)


if __name__ == "__main__":
	pt_dir = "outputs/single/pt/"
	ft_dir = "outputs/single/ft/"
	main(ft_dir)
