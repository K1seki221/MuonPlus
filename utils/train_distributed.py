import torch
import time 
import contextlib
import torch.distributed as dist
from utils.utils import get_worker_info, save_checkpoint, load_checkpoint
import json

typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}

class Logging():

    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []



def eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt):

    world_size, rank, local_rank, device  = get_worker_info()
    model.eval()
    val_loss = torch.tensor(0., device=device)
    counter = 0
    with torch.no_grad():
        for batch in val_dataloader:
            with autocast_ctxt:
                val_loss += torch.tensor(model(batch[0], batch[1], return_logits=False)[1], device=device)
            counter += 1
            if (val_accum_steps != 0) and (counter >= val_accum_steps): break
    val_loss = val_loss.detach().clone()/counter
    if world_size > 1: dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    if rank == 0:
        print(f"Validation Loss: {val_loss.item()}")
    model.train()
    return val_loss


def train(train_dataloader, val_dataloader, model, optimizer, training_params, logging_params, scheduler=None, ckpt_dir="", wandb_run=None):
    
    world_size, rank, local_rank, device  = get_worker_info()
    master_process = (rank == 0)
    logger = Logging()
    optimizer_name = optimizer.__class__.__name__
    if 'Momo' in optimizer_name:
        pass_loss = True
    else:
        pass_loss = False
    if master_process: print(f"Set pass_loss to {pass_loss} for optimizer {optimizer_name}")

    autocast_ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        autocast_ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']])     
    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (world_size*B*T))
    val_accum_steps = int(logging_params['val_tokens_processed'] / (world_size*B*T))
    if master_process: print(f"Accumulate gradient for {grad_accum_steps} steps")
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    max_grad_norm = training_params['gradnorm'] if training_params['gradnorm'] != 0. else float('inf')

    load_ckpt_step = logging_params['load_ckpt_step']
    if load_ckpt_step != 0:
        model, optimizer, train_dataloader, scheduler = load_checkpoint(ckpt_dir, load_ckpt_step, model, \
                                                        optimizer, train_dataloader, scheduler=scheduler)
    if ckpt_dir == "":
        print("Will not save checkpoints as no directory is specified")
    
    # Training loop
    for epoch in range(training_params['num_epochs']):
        if master_process:
            print(f"Epoch {epoch+1} of {training_params['num_epochs']}")

        model.train()
        start_epoch = time.time()
        start_time = time.time() 
        loss_accum = 0.
        step = 1 if load_ckpt_step == 0 else int(load_ckpt_step)
        optimizer.zero_grad()
        if step != 1: print(train_dataloader.get_state())
        
        for batch in train_dataloader:            
            with autocast_ctxt:
                loss = model(batch[0], batch[1], return_logits=False)[1]
                loss /= grad_accum_steps
            loss_accum += loss.detach()
                
            # Check if accummulated enough gradients to take a step
            if step % grad_accum_steps != 0:
                with (model.no_sync() if world_size > 1 else contextlib.nullcontext()):
                    loss.backward()
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if world_size > 1: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                if pass_loss:
                    optimizer.step(closure=None, loss=loss_accum)
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                    
                #bookkeeping
                torch.cuda.synchronize()
                step_time = time.time() - start_time
                if master_process and wandb_run is not None:
                    wandb_log_dict = {
                        "train/loss": loss_accum.item(),
                        "train/grad_norm": norm.item(),
                        "train/step_time": step_time,
                        "train/step": step
                    }
                    if hasattr(optimizer, 'step_size_list'):
                        wandb_log_dict["train/step_size_list"] = optimizer.step_size_list
                    # Log alpha value for svd-tempering with schedule
                    if hasattr(optimizer, 'current_alpha') and optimizer.current_alpha is not None:
                        wandb_log_dict["train/alpha"] = optimizer.current_alpha
                    if hasattr(optimizer, 'current_beta') and optimizer.current_beta is not None:
                        wandb_log_dict["train/spectral_beta"] = optimizer.current_beta
                    if hasattr(optimizer, 'current_ns') and optimizer.current_ns is not None:
                        wandb_log_dict["train/ns_steps"] = optimizer.current_ns
                    if hasattr(optimizer, 'current_p') and optimizer.current_p is not None:
                        wandb_log_dict["train/ns_softness_p"] = optimizer.current_p
                    if hasattr(optimizer, 'current_phase') and optimizer.current_phase is not None:
                        wandb_log_dict["train/optimizer_phase"] = optimizer.current_phase
                    if hasattr(optimizer, 'current_D') and optimizer.current_D is not None:
                        wandb_log_dict["train/kl_D"] = optimizer.current_D
                    if hasattr(optimizer, 'current_lambda') and optimizer.current_lambda is not None:
                        wandb_log_dict["train/kl_lambda"] = optimizer.current_lambda
                    for param_group_ix, param_group in enumerate(optimizer.param_groups):
                        wandb_log_dict[f"train/lr_{param_group_ix}"] = param_group['lr']
                    wandb_run.log(wandb_log_dict)
                logger.step_times.append(step_time)  # Are these different across ranks?
                logger.grad_norms.append(norm.item())
                for param_group in optimizer.param_groups:
                    logger.learning_rates.append(param_group['lr'])
                logger.losses.append(loss_accum.item())
                if hasattr(optimizer, 'step_size_list'):  
                    logger.step_size_list = optimizer.step_size_list  
                
                if (step % logging_params['log_step'] == 0) & master_process:
                    tps = training_params["tokens_processed"] / step_time
                    print(f"Step {step} of {total_iterations*grad_accum_steps}.")
                    print(f"Time taken : {step_time*1000:0.1f}ms | Tokens/s : {tps/1000:0.1f}k | Loss : {loss_accum.item():0.3f}")
                    
                if (step % logging_params['val_step'] == 0):
                    val_loss = eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt)
                    if master_process and wandb_run is not None:
                        wandb_run.log({"val/loss": val_loss.item(), "val/step": step})
                    logger.val_losses.append(val_loss.item())

                if (step % logging_params['save_ckpt_step'] == 0) & (ckpt_dir != ""):
                    save_checkpoint(ckpt_dir, step, model, optimizer, loss_accum.item(),
                                    train_dataloader, scheduler, logging_params['keep_last'])
                    
                    if master_process:
                        with open(ckpt_dir + '/log.json', 'w') as file:
                            json.dump(logger.__dict__, file)
                loss_accum = 0.
                start_time = time.time() 
            step += 1
            
            
        print(f"In rank: {rank}, epoch {epoch+1}, Train Loss: {logger.losses[-1]}")
        print(f"In rank: {rank}, time taken for epoch {epoch+1} : ", time.time() - start_epoch)
        
        # Evaluate on val set, and save final values
        val_dataloader.reset()
        val_loss = eval_validation_loss(model, val_dataloader, 0, autocast_ctxt)
        logger.val_losses.append(val_loss.item())
        print(f"In rank: {rank}, epoch {epoch+1}, Validation Loss: {val_loss.item()}")        
        if (ckpt_dir != ""):
            save_checkpoint(ckpt_dir, step, model, optimizer, logger.losses[-1],
                        train_dataloader, scheduler, logging_params['keep_last'])        
            if master_process:
                with open(ckpt_dir + '/log.json', 'w') as file:
                    json.dump(logger.__dict__, file)
        if master_process and wandb_run is not None:
            wandb_run.log({"val/loss": val_loss.item(), "val/step": step, "train/loss": logger.losses[-1], "train/step": step})

    if hasattr(optimizer, 'step_size_list'):      # Check if optimizer has a step_size_list attribute
        logger.step_size_list = optimizer.step_size_list  
    return logger
