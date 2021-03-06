// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
//   + BERT

local bert_model = "google/bert_uncased_L-8_H-512_A-8";
local max_length = 128;
local feature_size = 20;
local max_span_width = 10;

local bert_dim = 512;  # uniquely determined by bert_model
local lstm_dim = 200;
local span_embedding_dim = 4 * lstm_dim + bert_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

{
	"task_coref":{
		"task_description":{
			"task_name": "coref",
			"validation_metric_name": "coref_f1",
			"validation_metric_decreases": false,
			"evaluate_on_test": true
		},
		
		"data_params":{
			"dataset_reader": {
				"type": "coref_conll",
				"token_indexers": {
					"tokens": {
						"type": "pretrained_transformer_mismatched",
						"model_name": bert_model,
						"max_length": max_length
					},
				},
				"max_span_width": max_span_width
			},
			"train_data_path": "./data/conll-2012_single_file/train.english.gold_conll",
			"validation_data_path": "./data/conll-2012_single_file/dev.english.gold_conll",
			"test_data_path": "./data/conll-2012_single_file/test.english.gold_conll",
			"datasets_for_vocab_creation": ["train", "validation", "test"]
		}
	},

	"model": {
		"type": "coref_custom",
		"text_field_embedder": {
			"token_embedders": {
				"tokens": {
					"type": "pretrained_transformer_mismatched",
					"model_name": bert_model,
					"max_length": max_length
				}
			}
		},
		"coref": {
			"encoder": {
				"type": "lstm",
				"bidirectional": true,
				"hidden_size": lstm_dim,
				"input_size": bert_dim,
				"num_layers": 1
			},
			"tagger": {
				"mention_feedforward": {
					"input_dim": span_embedding_dim,
					"num_layers": 2,
					"hidden_dims": 150,
					"activations": "relu",
					"dropout": 0.2
				},
				"antecedent_feedforward": {
					"input_dim": span_pair_embedding_dim,
					"num_layers": 2,
					"hidden_dims": 150,
					"activations": "relu",
					"dropout": 0.2
				},
				"initializer": [
					[".*linear_layers.*weight", {"type": "xavier_normal"}],
					[".*scorer._module.weight", {"type": "xavier_normal"}],
					["_distance_embedding.weight", {"type": "xavier_normal"}],
					["_span_width_embedding.weight", {"type": "xavier_normal"}],
					["_context_layer._module.weight_ih.*", {"type": "xavier_normal"}],
					["_context_layer._module.weight_hh.*", {"type": "orthogonal"}]
				],
				"lexical_dropout": 0.5,
				"feature_size": feature_size,
				"max_span_width": max_span_width,
				"spans_per_word": 0.3,
				"max_antecedents": 100
			}
		}
	},
	
	"iterators": {
		"iterator": {
			"type": "basic",
			"batch_size": 32
		},
		"iterator_coref": {
			"type": "bucket",
			"sorting_keys": [["text", "tokens___token_ids"]],
			"padding_noise": 0.0,
			"batch_size": 1
		  }
	},
	
	"multi_task_trainer": {
		"type": "sampler_multi_task_trainer",
		"sampling_method": "proportional",
		"patience": 10,
		"num_epochs": 100,
		"min_lr": "1e-7",
		"grad_norm": 5.0,
		"grad_clipping": 10.0,
		"cuda_device": 0,
		"optimizer": {
			"type": "huggingface_adamw",
			"lr": 1e-3,
			"weight_decay": 0.01,
			"parameter_groups": [
				[[".*transformer.*"], {"lr": 1e-5}]]
		},
		"scheduler": {
			"type": "reduce_on_plateau",
			"factor": 0.5,
			"mode": "max",
			"patience": 2
		}
	}
}