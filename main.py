from lomo.src.utils import LearningRateScheduler,DynamicLossScaler, get_loss
from lomo.src.lomo import LOMO
import deepspeed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator



import os
import sys
import operator
from collections import OrderedDict
from itertools import chain
from pathlib import Path
import shutil
import torch.distributed as dist
import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DistributedSampler, DataLoader
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, SequentialDistributedSampler, nested_numpify
from transformers.trainer_utils import has_length, seed_worker
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, AutoConfig
from datasets import load_dataset


from data_processing import _prepare_packed_dataloader



model_name = "mistralai/Mistral-7B-v0.1"

dataset = load_dataset("HuggingFaceTB/cosmopedia-20k")


tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = _prepare_packed_dataloader(tokenizer=tokenizer,
                                           dataset = dataset["train"],
                                           dataset_text_field="prompt",
                                           max_seq_length=2048,
                                           num_of_sequences=1024,
                                           chars_per_token=3.6)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=2,collate_fn=data_collator)

config = AutoConfig.from_pretrained(model_name)
config.gradient_checkpointing = True

torch.set_default_dtype(torch.bfloat16)


model = AutoModelForCausalLM.from_pretrained(model_name,config=config,torch_dtype=torch.bfloat16,use_cache=False)

model, _, _, _ = deepspeed.initialize(
            config="ds_config.json",
            model=model,
        )


num_steps_per_epoch = len(train_dataloader)
global_step = 1
num_train_epochs = 1
per_device_train_batch_size=4
save_strategy = "steps"
save_total_limit=3
save_steps = 200
clip_grad_norm=1.0
clip_loss_value=None

local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()

n_steps = num_steps_per_epoch * num_train_epochs
lr_scheduler = LearningRateScheduler(learning_rate=2e-5,
                                            warmup=100,
                                            schedule="linear",
                                            n_steps=n_steps)
lr = 0

optimizer = LOMO(model, lr, clip_grad_norm=clip_grad_norm)

get_accelerator().empty_cache()

def train():
    for epoch in range(num_train_epochs):
        print(f"***** Running Training *****")
        print(f"  Num examples: {len(train_dataset)}")
        print(f"  Num Epochs: {num_train_epochs}")
        print(f"  Current Epoch: {epoch}")
        print(f"  Batch Size: {per_device_train_batch_size}")

        with tqdm.tqdm(train_dataloader) as tqb:
            for step, batch in enumerate(tqb, start=1):
                model.train()
                outs = model(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                )
                loss = get_loss(outs.logits, batch['labels'])

                # update the learning rate
                global_step = num_steps_per_epoch * epoch + step
                lr = lr_scheduler.step(global_step)
                if clip_grad_norm is not None and clip_grad_norm > 0:
                    optimizer.grad_norm(loss)
                    # gather_norm = True
                    # grad_norms = []
                    # loss_scaler.has_overflow_serial = False
                    # scaled_loss = loss * loss_scaler.loss_scale
                    #
                    # scaled_loss.backward()
                    # # update the last one since the hook function will not be called for the last parameter
                    # grad_func(0)

                    if optimizer.loss_scaler and optimizer.loss_scaler.has_overflow_serial:
                        print(f"Gradient overflow, skipping step {global_step}")
                        # loss_scaler.update_scale(overflow=True)
                        # with torch.no_grad():
                        #     for n, p in model.named_parameters():
                        #         p.grad = None
                        model.optimizer.get_param_coordinator(training=True).reset_step()
                        tqb.set_postfix({'loss': loss.item()})
                        continue

                    else:
                        model.optimizer.get_param_coordinator(training=True).reset_step()
                    # 第二次forward
                    outs = model(
                        input_ids=batch['input_ids'].cuda(),
                        attention_mask=batch['attention_mask'].cuda(),
                    )
                    loss = get_loss(outs.logits, batch['labels'], clip_loss_value)


                optimizer.fused_backward(loss, lr)


                

                model.optimizer.get_param_coordinator(training=True).reset_step()

                tqb.set_postfix({'loss': loss.item()})


                if save_strategy == 'steps' and global_step % save_steps == 0:
                    save_model(global_step)


        if save_strategy == 'epoch':
            save_model(epoch)

def save_model(index):
    if local_rank in [-1, 0]:
        checkpoint_dir = sorted(Path(output_dir).glob("checkpoint-*"))
        if len(checkpoint_dir) >= save_total_limit:
            shutil.rmtree(checkpoint_dir[0], ignore_errors=True)
    torch.distributed.barrier()

    output_dir = os.path.join(output_dir, f"checkpoint-{index}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    state_dict = OrderedDict() if torch.distributed.get_rank() == 0 else None
    shared_params = {}

    # Prepare for checkpoint save by ensuring all parameters are partitioned
    model.optimizer.partition_all_parameters()

    for name, param in model.module.named_parameters():
        with deepspeed.zero.GatheredParameters(param):
            if torch.distributed.get_rank() == 0:
                # can't rely on param.data_ptr() as it will be reused as weights gets
                # gathered and reduced, but param.ds_id is unique across all zero weights
                # (and shared params will have the same param.ds_id)
                if param.ds_id in shared_params:
                    # shared weights
                    state_dict[name] = state_dict[shared_params[param.ds_id]]
                else:
                    state_dict[name] = param.detach().cpu()
                    shared_params[param.ds_id] = name

    if len(model.optimizer.persistent_parameters) > 0:
        model.optimizer.persistent_parameters[0].all_gather(model.optimizer.persistent_parameters)

    if torch.distributed.get_rank() == 0:
        model.module.config.save_pretrained(output_dir)
        torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
        print(f"Save model to {output_dir}")

    torch.distributed.barrier()

train()