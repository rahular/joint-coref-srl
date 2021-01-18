local bert_model = "google/bert_uncased_L-2_H-128_A-2";
local max_length = 128;

local bert_dim = 128;  # uniquely determined by bert_model

{
	"task_srl":{
		"task_description":{
			"task_name": "srl",
			"validation_metric_name": "f1-measure-overall",
			"validation_metric_decreases": false,
			"evaluate_on_test": true
		},
		
		"data_params":{
			"dataset_reader": {
				"type": "srl_conll",
				"bert_model_name": bert_model
			},
		
			"train_data_path": "./data/conll-2012/v4/data/train",
			"validation_data_path": "./data/conll-2012/v4/data/development",
			"test_data_path": "./data/conll-2012/v4/data/test",

			"datasets_for_vocab_creation": ["train", "validation", "test"]
		}
	},
	
	"model": {
		"type": "srl_custom",
        "embedding_dropout": 0.1,
        "bert_model": bert_model,
		"label_smoothing": 0.1
	},
	
	"iterators": {
		"iterator": {
			"type": "basic",
			"batch_size": 32
		},
		"iterator_srl": {
            "type": "bucket",
			"sorting_keys": [['tokens', 'tokens___tokens']],
            "padding_noise": 0.0,
			"batch_size": 32
		}
	},
	
	"multi_task_trainer": {
		"type": "sampler_multi_task_trainer",
		"sampling_method": "proportional",
		"patience": 5,
		"num_epochs": 30,
		"grad_norm": 1.0,
		"grad_clipping": 10.0,
		"cuda_device": 0,
		"optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
            	[["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}]
			]
		},
		"scheduler": {
			"type": "reduce_on_plateau", 
			"mode": "max", 
			"factor": 0.5,
			"patience": 2,
			"verbose": true
		}
	}
}