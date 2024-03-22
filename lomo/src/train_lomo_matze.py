import os
import sys
import operator
from collections import OrderedDict
from itertools import chain
from pathlib import Path
import shutil
import torch
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed
from dataclasses import asdict
from transformers.deepspeed import HfDeepSpeedConfig
import wandb
import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DistributedSampler, DataLoader
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, SequentialDistributedSampler, nested_numpify
from transformers.trainer_utils import has_length, seed_worker
from transformers import GenerationConfig
from log import print
from arguments import ModelArguments, DataArguments, MyTrainingArguments
from mydatasets import MyDataset, get_dataset_info
from lomo_trainer import LOMOTrainer
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM

try:
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
except:
    pass

from src.utils import LearningRateScheduler, WandbLogger, DynamicLossScaler, get_loss
from src.lomo import LOMO
from log import print

torch.set_default_dtype(torch.float16)
parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
if sys.argv[-1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
set_seed(training_args.seed)

model_name = "Phoenix"


ds_config = training_args.deepspeed
dschf = HfDeepSpeedConfig(ds_config)
config = AutoConfig.from_pretrained(model_args.model_name_or_path)
config.gradient_checkpointing = training_args.gradient_checkpointing

model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id




## Arguments


train_dataset = ""
eval_dataset = ""


data_collator=""

training_args = "dict()"


wandb = WandbLogger(training_args)
allow_print = training_args.local_rank in [-1, 0]

if 'deepspeed' not in sys.modules:
            raise ModuleNotFoundError(
                "Detected DeepSpeed is not installed. See https://github.com/microsoft/DeepSpeed")

# Initialize deepspeed engine
model, _, _, _ = deepspeed.initialize(
    config=training_args.deepspeed,
    model=model,
)

        

num_steps_per_epoch = len(train_dataloader)
global_step = 1
n_steps = num_steps_per_epoch * training_args.num_train_epochs
lr_scheduler = LearningRateScheduler(learning_rate=training_args.learning_rate,
                                            warmup=training_args.warmup,
                                            schedule=training_args.lr_scheduler_type,
                                            n_steps=n_steps)
lr = 0

optimizer = LOMO(model, lr, training_args.clip_grad_norm, training_args.clip_grad_value)

get_accelerator().empty_cache()

def train(self):
    for epoch in range(training_args.num_train_epochs):
        print(f"***** Running Training *****")
        print(f"  Num examples: {len(train_dataset)}")
        print(f"  Num Epochs: {training_args.num_train_epochs}")
        print(f"  Current Epoch: {epoch}")
        print(f"  Batch Size: {training_args.per_device_train_batch_size}")
        if allow_print:
            wandb.log({'train/epoch': epoch}, step=global_step)
        train_dataloader.sampler.set_epoch(epoch)

        with tqdm.tqdm(train_dataloader, disable=not allow_print) as tqb:
            for step, batch in enumerate(tqb, start=1):
                model.train()
                outs = model(
                    input_ids=batch['input_ids'].cuda(),
                    attention_mask=batch['attention_mask'].cuda(),
                )
                loss = get_loss(outs.logits, batch['labels'], training_args.clip_loss_value)

                # update the learning rate
                global_step = num_steps_per_epoch * epoch + step
                lr = lr_scheduler.step(global_step)
                if training_args.clip_grad_norm is not None and training_args.clip_grad_norm > 0:
                    optimizer.grad_norm(loss)


                    if optimizer.loss_scaler and optimizer.loss_scaler.has_overflow_serial:
                        print(f"Gradient overflow, skipping step {global_step}")

                        model.optimizer.get_param_coordinator(training=True).reset_step()
                        tqb.set_postfix({'loss': loss.item()})
                        if allow_print:
                            wandb.log(
                                {
                                    'train/loss': loss.item(),
                                    'train/learning_rate': lr,
                                    'train/global_step': global_step,
                                },
                                step=global_step
                            )
                        continue

                    else:
                        model.optimizer.get_param_coordinator(training=True).reset_step()
                    # 第二次forward
                    outs = model(
                        input_ids=batch['input_ids'].cuda(),
                        attention_mask=batch['attention_mask'].cuda(),
                    )
                    loss = get_loss(outs.logits, batch['labels'], training_args.clip_loss_value)

                optimizer.fused_backward(loss, lr)
                model.optimizer.get_param_coordinator(training=True).reset_step()

                tqb.set_postfix({'loss': loss.item()})
                if allow_print:
                    wandb.log(
                        {
                            'train/loss': loss.item(),
                            'train/learning_rate': lr,
                            'train/global_step': global_step,
                        },
                        step=global_step
                    )

                if training_args.save_strategy == 'steps' and global_step % training_args.save_steps == 0:
                    save_model(global_step)



        if training_args.save_strategy == 'epoch':
            save_model(epoch)




def get_train_sampler(self):
    if train_dataset is None or not has_length(train_dataset):
        return None

    # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
    # `training_args.seed`) if data_seed isn't provided.
    # Further on in this method, we default to `training_args.seed` instead.
    seed = training_args.data_seed if training_args.data_seed is not None else training_args.seed

    if training_args.group_by_length:
        return DistributedLengthGroupedSampler(
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            dataset=train_dataset,
            num_replicas=training_args.world_size,
            rank=training_args.local_rank,
            lengths=None,
            model_input_name="input_ids",
            seed=seed,
        )
    else:
        return DistributedSampler(
            train_dataset,
            num_replicas=training_args.world_size,
            rank=training_args.local_rank,
            seed=seed
        )

def get_train_dataloader(self):
    """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
    if train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    data_collator = train_data_collator
    train_sampler = get_train_sampler()

    return DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=training_args.dataloader_drop_last,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
        worker_init_fn=seed_worker,
    )


def save_model(self, index):
    if self.training_args.local_rank in [-1, 0]:
        checkpoint_dir = sorted(Path(self.training_args.output_dir).glob("checkpoint-*"))
        if len(checkpoint_dir) >= self.training_args.save_total_limit:
            shutil.rmtree(checkpoint_dir[0], ignore_errors=True)
    torch.distributed.barrier()

    output_dir = os.path.join(self.training_args.output_dir, f"checkpoint-{index}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    state_dict = OrderedDict() if torch.distributed.get_rank() == 0 else None
    shared_params = {}

    # Prepare for checkpoint save by ensuring all parameters are partitioned
    self.model.optimizer.partition_all_parameters()

    for name, param in self.model.module.named_parameters():
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

    if len(self.model.optimizer.persistent_parameters) > 0:
        self.model.optimizer.persistent_parameters[0].all_gather(self.model.optimizer.persistent_parameters)

    if torch.distributed.get_rank() == 0:
        self.model.module.config.save_pretrained(output_dir)
        torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
        print(f"Save model to {output_dir}")

    torch.distributed.barrier()
