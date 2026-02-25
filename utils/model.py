from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AutoTokenizer, AutoModelForCausalLM
from .models import GPT, GPTConfig, LlamaModel, LlamaConfig

def load_model_and_tokenizer(config, device):

    if 'model_name' in config['model_config']:
        print(f"Loading model {config['model_config']['model_name']} and ignoring other configurations if specified")
        model = AutoModelForCausalLM.from_pretrained(config['model_config']['model_name'], device_map="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(config['model_config']['model_name'])
        if not config['model_config']['pretrained']:
            model_config = model.config
            del model
            model = GPT2LMHeadModel(model_config).to(device)
        else:
            print("Using pre-trained version")
            
    else:
        gpt_config = config['model_config']
        model_config = GPT2Config(
            n_embd=gpt_config['n_embd'],   
            n_layer=gpt_config['n_layer'],  
            n_head=gpt_config['n_head'],    
            vocab_size=gpt_config['vocab_size'], 
        )
        model = GPT2LMHeadModel(model_config).to(device)   # Initialize a new model with random weights using this configuration
        print("Loading gpt2 tokenizer as default tokenizer\n")
        tokenizer = AutoTokenizer.from_pretrained(config['model_config']['gpt2'])
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_huggingface(config, device):

    if 'model_name' in config['model_config']:
        print(f"Loading model {config['model_config']['model_name']} and ignoring other configurations if specified")
        model = AutoModelForCausalLM.from_pretrained(config['model_config']['model_name'], device_map="auto")#.to(device)
        if not config['model_config']['pretrained']:
            model_config = model.config
            del model
            model = GPT2LMHeadModel(model_config)#.to(device)
        else:
            print("Using pre-trained version")
            
    else:
        gpt_config = config['model_config']
        model_config = GPT2Config(
            n_embd=gpt_config['n_embd'],   
            n_layer=gpt_config['n_layer'],  
            n_head=gpt_config['n_head'],    
            vocab_size=gpt_config['vocab_size'], 
        )
        model = GPT2LMHeadModel(model_config)#.to(device)   # Initialize a new model with random weights using this configuration
    return model


def load_model(config, device):
    model_type = config.get('model_type', 'gpt')  # 默认为 'gpt'

    if model_type == 'llama':
        llamaconfig = LlamaConfig()
        llamaconfig.hidden_size = config['n_embd']
        llamaconfig.num_hidden_layers = config['n_layer']
        llamaconfig.num_attention_heads = config['n_head']
        llamaconfig.num_key_value_heads = config.get('n_kv_head', config['n_head'])
        llamaconfig.vocab_size = config['vocab_size']
        llamaconfig.max_position_embeddings = config.get('max_position_embeddings', config.get('block_size', 4096))
        if 'intermediate_size' in config:
            llamaconfig.intermediate_size = config['intermediate_size']
        if 'rope_theta' in config:
            llamaconfig.rope_theta = config['rope_theta']
        if 'rms_norm_eps' in config:
            llamaconfig.rms_norm_eps = config['rms_norm_eps']
        model = LlamaModel(llamaconfig, device, flash_attention=config['flash_attention'])
    else:
        gptconfig = GPTConfig()
        gptconfig.n_embd = config['n_embd']
        gptconfig.n_layer = config['n_layer']
        gptconfig.n_head = config['n_head']
        gptconfig.vocab_size = config['vocab_size']
        gptconfig.block_size = config['block_size']
        model = GPT(gptconfig, device, flash_attention=config['flash_attention'])
    return model

