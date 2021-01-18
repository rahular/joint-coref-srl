# coref-srl-rl

Code to reproduce experiments from "Joint Semantic Analysis with Document-Level Cross-Task Coherence Rewards"

*Recommended: Python 3.7 with a new conda environment*

## Installation

Install all dependencies with `pip install -r requirements.txt`

Then navigate to the `allennlp` folder and run the following commands
```
$ pip install --editable .
$ pip install -r dev-requirements.txt
```

Install `pytorc-geometric` by running `scripts/install_pytg.sh`. If that fails, follow [this](https://github.com/rusty1s/pytorch_geometric/issues/1001#issuecomment-598708757)

## Training

- Download Ontonotes 5.0 and run `prepare_ontonotes.sh` to convert it into the CoNLL-2012 format
- Make sure that the concatenated files are in a folder named `data/conll-2012_single_file` with names like `test.english.gold_conll` (without the version number) and the original split files are in a folder named `data/conll-2012`
- Run `mkdir graphs && python combine_data.py` to combine the gold (train and  development) SRL tags and corefernce clusters into a single JSON object. This should create 2 files, one for training and development each
- To further convert them into processable  graphs, run  `python graph_util/output_to_graph.py <JSON-path-of-graphs>` (once for train and again for dev.)
- Finally, train the coherence classifiers by running `dgi.py`

Now you are all set to train the coreference and SRL models!

All the configuration files can be found in the `configs` folder:
* `single`: contains the configs for the single-task baselines
* `mtl`: contains the configs for the multi-task baselines
* `ft`: contains the configs to finetune the different models (each encoder type has a different config file)

To train the baseline models, run
```
$ python train.py --config_file_path <path-to-config> --serialization_dir <path-to-save>
```

To finetune the models, run
```
$ python predict.py <path-to-ft-config> <reward-type> <graph-encoder-type> <ft-task-name> <ft-dataset> <path-to-model-dir>
```

In all our experiments we use `reward-type=sep` and `graph-encoder-type=GCN`.

## Evaluation

To evaluate a model on a dataset, run

```
$ python evaluate.py -s <path-to-model-dir> -t <task-name> -d <dataset-name> -m <type-of-model>
```

`type-of-model` can be `pt` or  `ft` for evaluating pre-trained and fine-tuned models respectively.

For all files, running it with `--help` flag prints the various options and their help strings.

## Thanks
This code is built on [HMTL](https://github.com/huggingface/hmtl)

## Citation
```
@misc{aralikatte2020joint,
      title={Joint Semantic Analysis with Document-Level Cross-Task Coherence Rewards}, 
      author={Rahul Aralikatte and Mostafa Abdou and Heather Lent and Daniel Hershcovich and Anders Søgaard},
      year={2020},
      eprint={2010.05567},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```