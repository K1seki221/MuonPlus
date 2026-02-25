import numpy as np
import torch
import random
import yaml
import hashlib
import json
import torch.distributed as dist
import os

# get worker info for distributed training
def get_worker_info():
    if dist.is_initialized():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return world_size, rank, local_rank, device
        

def hash_config(optimizer_config, training_params, model_config):
    """
    Generate a hash from the relevant fields of the current optimizer configuration,
    training parameters, and model configuration.

    Parameters
    ----------
    optimizer_config : dict
        The configuration dictionary for the current optimizer.
    training_params : dict
        The training parameters dictionary.
    model_config : dict
        The model configuration dictionary.

    Returns
    -------
    str
        A compressed hash string.
    """
    # Combine relevant fields
    relevant_fields = {
        "optimizer_config": optimizer_config,
        "training_params": training_params,
        "model_config": model_config
    }
    # Convert to a JSON string and hash it
    config_str = json.dumps(relevant_fields, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility

def compute_cross_entropy_loss(model, input_ids, attention_mask, labels):
    # Get model outputs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
    # Shift the logits and labels for language modeling
    shift_logits = logits[:, :-1, :].contiguous()  # Remove the last token's logits
    shift_labels = labels[:, 1:].contiguous()      # Remove the first token in the labels
    
    # Flatten the logits and labels for cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding token (assumes -100 for padding)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss

def get_outputfile_from_configfile(config_file):
    return 'gptopt/outputs/' + config_file.replace('configs/', '', 1).replace('.yaml', '', 1) + '.json'

def get_default_config():
    default_config = {
        "optimizer_params": [
        {
            "name": "adam",
            "lr": [ 0.0001],
            "weight_decay": 0,
            "lr_schedule": "constant"
        },
        {
            "name": "momo-adam",
            "lr": [ 0.1],
            "weight_decay": 0,
            "lr_schedule": "constant"
        },
        {
            "name": "sgd-m",
            "lr": [0.001],
            "weight_decay": 0,
            "momentum": 0.9,
            "dampening": 0.9,
            "lr_schedule": "warm-up-cosine",
            "warm_up_percent": 0.2
        }
    ],
    "training_params": {
        "batch_size": 8,
        "num_epochs": 1,
        "max_length": 512
    },
    "model_config": {
        "model_name": "gpt2",  # You can use one of the pre-defined models of transformers, or you can specify the exact dimension below
        "n_embd": 768,  # Hidden size used in distilgpt2
        "n_layer": 12,  # Number of layers in distilgpt2
        "n_head": 12,  # Number of attention heads in distilgpt2
        "vocab_size": 50257,
        "tokenizer_name": "gpt2"
    },
        'dataset': {
            'name': 'wikitext-2-raw-v1',
            'problem_name': 'default'
        }
    }
    return default_config

# Function to recursively merge dictionaries
def merge_configs(default_config, user_config):
    for key, value in default_config.items():
        if key not in user_config:
            user_config[key] = value
        elif isinstance(value, dict) and isinstance(user_config[key], dict):
            merge_configs(value, user_config[key])
    return user_config

def load_config(default_config, config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return merge_configs(default_config, config)



def save_checkpoint(ckpt_dir, step, model, optimizer, loss, dataloader, scheduler=None, keep_last=2):
    world_size = dataloader.world_size
    master_process = (dataloader.rank == 0)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Gather dataloader from all ranks and save it.
    if world_size > 1:
        dataloader_states = [None for _ in range(world_size)]
        state = dataloader.get_state()
        dist.all_gather_object(dataloader_states, state)
    else:
        dataloader_states = [dataloader.get_state()]
    
    # Save the checkpoint and daaloader to a file (e.g. checkpoint.pth)
    if master_process:
        print("Save checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(checkpoint, ckpt_dir + f'/ckpt{step}.pth')
        with open(ckpt_dir + f'/ckpt{step}_loss.json', "w") as f:
            json.dump({'train_loss':loss}, f)

        with open(ckpt_dir + f'/ckpt{step}_dataloader.json', "w") as f:
            json.dump(dataloader_states, f, indent=4)

        # Delete old checkpoints
        manage_checkpoint_directory(ckpt_dir, keep_last)
    

def load_checkpoint(ckpt_dir, step, model, optimizer, dataloader, scheduler=None):
    # Assume the model and optimizer are already created (they should have the same structure)
    print(f"Loading checkpoint {ckpt_dir + f'/ckpt{step}.pth'}")
    # Load to CPU first to avoid doubling GPU memory during restore
    checkpoint = torch.load(ckpt_dir + f'/ckpt{step}.pth', map_location='cpu')

    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint['model_state_dict']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    del checkpoint['optimizer_state_dict']
    # Move optimizer states to the correct device
    _, _, _, device = get_worker_info()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    torch.cuda.empty_cache()
    if scheduler is not None:
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print("WARNING: Scheduler dict not found in checkpoint")
    
    print(f"Loss at checkpoint : {checkpoint['loss']}")
    with open(ckpt_dir + f'/ckpt{step}_dataloader.json', "r") as f:
        dataloader_states = json.load(f)

    for state in dataloader_states:
        if state['rank'] == dataloader.rank:
            dataloader.set_state(state)

    return model, optimizer, dataloader, scheduler


def manage_checkpoint_directory(ckpt_dir, keep_last=2):
    # List checkpoint files (assuming .pth extension)
    ckpt_files = [os.path.join(ckpt_dir, f)
                        for f in os.listdir(ckpt_dir)
                        if f.endswith(".pth")]
    loss_files = [os.path.join(ckpt_dir, f)
                        for f in os.listdir(ckpt_dir)
                        if f.endswith("_loss.json")]
    
    if not ckpt_files:
        print("No ckpt files found.")
        return
                            
    ckpt_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    recent_ckpts = ckpt_files[:keep_last]

    best_loss = float("inf")
    best_ckpt = None
    
    for file in loss_files:
        try:
            with open(file, "r") as f:
                loss = json.load(f)['train_loss']
            if loss < best_loss:
                best_loss = loss
                best_ckpt = file.split("_loss")[0] + ".pth"
        except Exception as e:
            print(f"Could not load loss from {file}: {e}")

    # Build a set of files to keep: recent ones plus the best-loss one.
    to_keep = set(recent_ckpts)
    if best_ckpt is not None:
        to_keep.add(best_ckpt)
    
    # Delete any ckpt file not in the set to_keep.
    for file in ckpt_files:
        if file not in to_keep:
            os.remove(file)
