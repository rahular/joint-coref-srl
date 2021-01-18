"""
Combines coref and srl data and dumps into a JSON
The output of this file becomes the input for graph construction
"""

import json
import pickle as pkl

from uuid import uuid4
from collections import defaultdict
from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes

from allennlp.data.dataset_readers.coreference_resolution.conll import ConllCorefReader
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader

from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer


def _normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def _find_index(doc, flattened_ontonotes):
    try:
        return flattened_ontonotes.index(doc)
    except ValueError:
        return -1


def read_raw_ontonotes_docs(path):
    ontonotes, flattened_ontonotes = [], []
    ontonotes_reader = Ontonotes()
    for sentences in ontonotes_reader.dataset_document_iterator(path):
        ontonotes.append([s.words for s in sentences])
    for doc in ontonotes:
        flattened_ontonotes.append(
            [_normalize_word(word) for sentence in doc for word in sentence]
        )

    assert len(ontonotes) == len(flattened_ontonotes)
    return ontonotes, flattened_ontonotes


def read_coref_ontonotes(path):
    indexers = {
        "tokens": SingleIdTokenIndexer(lowercase_tokens=True),
        "token_characters": TokenCharactersIndexer(),
    }
    reader = ConllCorefReader(max_span_width=10, token_indexers=indexers)
    data = reader.read(path)

    cdata = []
    for dp in data:
        obj = dp.fields["metadata"].metadata
        # reformat to be consistent with the predictions
        obj["document"] = [obj["original_text"]]
        obj["clusters"] = [obj["clusters"]]
        del obj["original_text"]
        cdata.append(obj)
    return cdata


def read_srl_ontonotes(path):
    indexers = {
        "tokens": SingleIdTokenIndexer(lowercase_tokens=True),
        "token_characters": TokenCharactersIndexer(),
    }
    reader = SrlReader(token_indexers=indexers)
    data = reader.read(path)

    sdata = []
    no_verb = 0
    for dp in data:
        obj = dp.fields["metadata"].metadata
        if not obj["verb"]:
            no_verb += 1
            continue
        # reformat to be consistent with the predictions
        obj["tags"] = [dp.fields["tags"].labels]
        obj["words"] = [obj["words"]]
        obj["verb"] = [obj["verb"]]
        sdata.append(obj)
    return sdata, no_verb


def read_pkl(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def write_pkl(data, path):
    with open(path, "wb") as f:
        pkl.dump(data, f)


def make_srl_dict(sdata):
    srl_dict = defaultdict(list)
    for s in sdata:
        for words, verb, tags in zip(s["words"], s["verb"], s["tags"]):
            srl_dict[" ".join(words)].append({"verb": verb, "tags": tags})
    return srl_dict


def combine(cdata, srl_dict, part="test"):
    output = {}
    total, total_missed = 0, 0

    # read original Ontonotes with sentence demarcations
    ontonotes, flattened_ontonotes = read_raw_ontonotes_docs(
        "./data/conll-2012_single_file/{}.english.gold_conll".format(part)
    )

    for dnum, c in enumerate(cdata):
        num_sents, missed = 0, 0
        obj = {}
        doc_idx = _find_index(c["document"][0], flattened_ontonotes)
        if doc_idx > -1:
            obj["document"] = flattened_ontonotes[doc_idx]
            obj["clusters"] = c["clusters"]
            obj["sentences"] = []
            obj["srl_tags"] = []
            for sent in ontonotes[doc_idx]:
                srl_obj = srl_dict.get(" ".join(sent), None)
                obj["sentences"].append(sent)
                obj["srl_tags"].append(srl_obj)
                if not srl_obj:
                    missed += 1
                num_sents += 1
            output[str(uuid4())] = obj
            # print('Doc {} missed {}/{} sentences...'.format(dnum+1, missed, num_sents))
        total += num_sents
        total_missed += missed
    print("Total missed sentences: {}/{}".format(total_missed, total))
    return output


def save(output, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f)


def index_span_pkl(data, sdata):
    found = 0
    new_sdata = {}
    for key, val in data.items():
        for s in sdata:
            if s["document"][0] == val["document"]:
                new_sdata[key] = s
                found += 1
                break
    assert found == len(sdata)
    return new_sdata


def main():
    print('Combining training data...')
    cdata = read_coref_ontonotes('./data/conll-2012_single_file/train.english.gold_conll')
    sdata, no_verb = read_srl_ontonotes('./data/conll-2012/v4/data/train/')
    srl_dict = make_srl_dict(sdata)
    output = combine(cdata, srl_dict, part='train')
    assert len(output) == len(cdata)
    save(output, './graphs/combined_train.json')
    print('Sentences without verbs: {}'.format(no_verb))

    print('Combining validation data...')
    cdata = read_coref_ontonotes('./data/conll-2012_single_file/dev.english.gold_conll')
    sdata, no_verb = read_srl_ontonotes('./data/conll-2012/v4/data/development/')
    srl_dict = make_srl_dict(sdata)
    output = combine(cdata, srl_dict, part='dev')
    assert len(output) == len(cdata)
    save(output, './graphs/combined_dev.json')
    print('Sentences without verbs: {}'.format(no_verb))


if __name__ == "__main__":
    main()
