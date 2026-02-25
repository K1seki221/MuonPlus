import yaml
import argparse
import torch
import matplotlib.pyplot as plt
from utils.train_distributed import train
from utils.optim.utils import get_scheduler, get_optimizer
from utils.utils import hash_config, set_seed, get_worker_info
#from utils.utils import get_default_config, load_config
from utils.model import load_model
from utils.dataloader import DATA_DIR, ShardedDataLoader, get_dataset_path
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy 
import json
import os
import wandb

parser = argparse.ArgumentParser(description='Training with optional config file.')
parser.add_argument('--config', type=str, help='Path to config file', default=None)
parser.add_argument('--suffix', type=str, help='Path to config file', default='')
parser.add_argument('--prefix', type=str, help='Path to config file', default='')
args = parser.parse_args()
set_seed(42)

# First set up DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    dist.init_process_group(backend='nccl')
world_size, rank, local_rank, device = get_worker_info()
master_process = (rank == 0) # this process will do logging, checkpointing etc.
device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f"Using device: {device}")

# Parse config
config_file = args.config
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
#config = load_config(get_default_config(), config_file)
outputname = config_file.replace("configs/","").replace('.yaml','')
output_dir = f"{config['logging_params']['results_dir']}/{outputname}"
CKPT_DIR = config['logging_params']['ckpt_dir']
ckpt_dir_base = CKPT_DIR + f"/{outputname}/" if CKPT_DIR != "" else ""
if master_process:
    print(f"Loading configuration from {config_file}")
    print(f"Training on dataset {config['dataset']['name']}")
    os.makedirs(output_dir, exist_ok=True)  
    if CKPT_DIR != "": os.makedirs(ckpt_dir_base, exist_ok=True)  

# Load model
model = load_model(config['model_config'], device)

# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if master_process:
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

print(model)
# Set the training parameters
training_params = config['training_params'] 
list_optimizer_params = config["optimizer_params"]
torch.set_float32_matmul_precision(training_params['tensorcore_precision'])

# Load data
tokenizer_name = config['dataset'].get('tokenizer', 'gpt2')
dataset_path = get_dataset_path(config['dataset']['name'], tokenizer_name)
if master_process: print(f"Load data from {dataset_path}")
B, T = training_params['batch_size'], training_params['context_length']
assert training_params['tokens_processed'] % (world_size * B * T) == 0
# Calculate number of shards based on total_tokens_B (each shard is ~0.1B tokens)
num_shards = None
if 'total_tokens_B' in config['dataset']:
    num_shards = int(config['dataset']['total_tokens_B'] * 10)  # 0.1B per shard
    if master_process: print(f"Limiting training to {config['dataset']['total_tokens_B']}B tokens ({num_shards} shards)")
train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device, num_shards=num_shards)
val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
if master_process:
    print(f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens")
    print(f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens")
    print(f"Total number of iterations : {total_iterations}")

# Loop over optimizers
for opt_config in list_optimizer_params:
    for lr in opt_config['lr']:
        print()
        if master_process:
            print(f"Training with optimizer {opt_config['name']} and learning rate {lr}")
            
        # Generate hash for the current optimizer configuration
        opt_config_copy = copy.deepcopy(opt_config)
        opt_config_copy['lr'] = lr
        config_hash = hash_config(opt_config_copy, training_params, config['model_config'])
        file_name = f"{opt_config['name']}-lr-{lr}-{opt_config['lr_schedule']}-{config_hash}-world{world_size}"
        if args.suffix != '': file_name += f"-{args.suffix}"
        output_path = os.path.join(output_dir, file_name + '.json')
        ckpt_dir = os.path.join(ckpt_dir_base, file_name) + '/' if CKPT_DIR != "" else ""
        
        # copy model to ensure consistency
        model_copy = copy.deepcopy(model).to(device)
        if training_params['compile']:
            if master_process: print("Compiling model")
            model_copy = torch.compile(model_copy)
        if ddp:
            model_copy = DDP(model_copy, device_ids=[local_rank])

        # Setup optimizer
        optimizer_obj, hyperp = get_optimizer(opt_config, lr=lr)
        p = model_copy.named_parameters() if 'muon' in opt_config['name'] else model_copy.parameters()
        optimizer = optimizer_obj(p, **hyperp)
        scheduler = get_scheduler(opt_config, optimizer, total_iterations=total_iterations)

        # Initialize wandb
        if master_process and config['logging_params'].get('wandb', None) is not None:
            config_no_optimizer = copy.deepcopy(config)
            del config_no_optimizer['optimizer_params']
            wandb_config = dict(one_optimizer_params=opt_config_copy, **config_no_optimizer, world_size=world_size)
            if "dir" not in config['logging_params']['wandb']:
                config['logging_params']['wandb']['dir'] = f"{config['logging_params']['ckpt_dir']}/../wandb"
            wandb_run = wandb.init(
                **config['logging_params']['wandb'],
                config=wandb_config,
                name = args.prefix+file_name,
                reinit=True,
            )
        else:
            wandb_run = None

        # Train
        try:
            logger = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params,
                        scheduler=scheduler, ckpt_dir=ckpt_dir,
                        logging_params=config['logging_params'], wandb_run=wandb_run)
        finally:
            if master_process and wandb_run is not None:
                wandb_run.finish()

        # Save
        if master_process:
            if 'muon-compact' in opt_config['name']:
                polar_params = opt_config['polar_params']   
                logger.name = opt_config['name'] + '-pin-' +str(polar_params['pinpoint_top'])
                logger.name = logger.name + '-ns-' +str(opt_config['ns_steps'])
                logger.name = logger.name  +'-fa-' +str(polar_params['fast_apply_restart'])
                logger.name = logger.name  +'-defl-' +str(polar_params['deflation_eps'])
                logger.name = logger.name  +'-lr-' + str(lr)
            else:
                logger.name = opt_config['name'] + '-lr-' + str(lr)

            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Overwriting")
            with open(output_path, 'w') as file:
                json.dump(logger.__dict__, file)
            print(f"Saved output to {output_path}")

if ddp:
    dist.destroy_process_group()
