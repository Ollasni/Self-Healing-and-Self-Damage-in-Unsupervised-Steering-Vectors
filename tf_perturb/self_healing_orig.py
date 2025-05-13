import os
import torch
from torch import Tensor, Union
from tqdm import tqdm
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional, Literal

from tf_perturb.direct_effect import (
    collect_direct_effect,
    get_correct_logit_score,
    topk_of_Nd_tensor,
)
import transformer_lens as ts
from transformer_lens import HookedTransformer, ActivationCache

from tf_perturb.dataset import prepare_dataset
from tf_perturb.path_patching import Node, act_patch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "data/figures/breakdown_self_repair_graphs/"
FOLDER_TO_STORE_PICKLES = "data/pickle_storage/breakdown_self_repair/"

# check existence of folders
if not os.path.exists(FOLDER_TO_WRITE_GRAPHS_TO):
    os.makedirs(FOLDER_TO_WRITE_GRAPHS_TO)
if not os.path.exists(FOLDER_TO_STORE_PICKLES):
    os.makedirs(FOLDER_TO_STORE_PICKLES)


# model_name = "gpt2-small"  # "pythia-160m"
# BATCH_SIZE = 2
# PERCENTILE = 0.02
# MIN_TOKENS = 1_000


# model = ts.HookedTransformer.from_pretrained(
#     model_name,
#     center_unembed=True,
#     center_writing_weights=True,
#     fold_ln=True,  # TODO; understand this
#     refactor_factored_attn_matrices=False,
#     device=device,
# )

# safe_model_name = model_name.replace("/", "_")
# model.set_use_attn_result(True)

# dataset = ts.utils.get_dataset("pile")
# dataset_name = "The Pile"
# all_dataset_tokens = model.to_tokens(dataset["text"]).to(device)
 

def return_item(item):
    return item


def ablate_top_instances_and_get_breakdown(
    model: HookedTransformer,
    head: tuple,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    per_head_direct_effect: Union[Tensor, None] = None,
    all_layer_direct_effect: Union[Tensor, None] = None,
    cache: Union[ts.ActivationCache, None] = None,
    logits: Union[Tensor, None] = None,
):

    if cache is None or logits is None:
        print("Cache not provided")
        cache, logits = model.run_with_cache(clean_tokens)
        assert cache is not None
        assert logits is not None

    if per_head_direct_effect is None or all_layer_direct_effect is None:
        print("Per head direct effect not provided")
        per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(
            cache,
            correct_tokens=clean_tokens,
            model=model,
            display=False,
            collect_individual_neurons=False,
            cache_for_scaling=cache,
        )
        assert per_head_direct_effect is not None
        assert all_layer_direct_effect is not None

    # Run ablation and get new cache/logit/DEs
    ablated_cache: ts.ActivationCache = act_patch(
        model, clean_tokens, [Node("z", head[0], head[1])], 
        return_item, corrupted_tokens, apply_metric_to_cache=True
    )  # type: ignore
    ablated_logits = act_patch(
        model,
        clean_tokens,
        [Node("z", head[0], head[1])],
        return_item,
        corrupted_tokens,
        apply_metric_to_cache=False,
    )
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = (
        collect_direct_effect(
            ablated_cache,
            correct_tokens=clean_tokens,
            model=model,
            display=False,
            collect_individual_neurons=False,
            cache_for_scaling=cache,
        )
    )

    # get the logit difference between everything
    correct_logits = get_correct_logit_score(logits, clean_tokens)
    ablated_logits = get_correct_logit_score(ablated_logits, clean_tokens)
    logit_diffs = ablated_logits - correct_logits

    # Get Direct Effect of Heads Pre/Post-Ablation
    direct_effects = per_head_direct_effect[head[0], head[1]]
    ablated_direct_effects = ablated_per_head_direct_effect[head[0], head[1]]

    # Calculate self-repair values
    self_repair, self_repair_from_heads = get_self_repair_all(head, per_head_direct_effect, ablated_per_head_direct_effect, logit_diffs, direct_effects, ablated_direct_effects)

    return (
        logit_diffs,
        direct_effects,
        ablated_direct_effects,
        self_repair_from_heads,
        self_repair_from_layers,
        self_repair_from_LN,
        self_repair,
    )
def calculate_self_repair(logit_diffs, direct_effects, ablated_direct_effects):
    """Calculate the total self-repair effect."""
    return logit_diffs - (ablated_direct_effects - direct_effects)

def calculate_self_repair_from_heads(head, per_head_direct_effect, ablated_per_head_direct_effect):
    """Calculate the self-repair contribution from attention heads."""
    abladed_de_minus_de = (ablated_per_head_direct_effect - per_head_direct_effect).sum((0, 1))
    ablated_de_minus_de_per_head = (ablated_per_head_direct_effect - per_head_direct_effect)[head[0], head[1]]
    return abladed_de_minus_de - ablated_de_minus_de_per_head

def calculate_self_repair_from_layers(all_layer_direct_effect, ablated_all_layer_direct_effect):
    """Calculate the self-repair contribution from MLP layers."""
    return (ablated_all_layer_direct_effect - all_layer_direct_effect).sum(0)


def calculate_self_repair_from_LN(self_repair, self_repair_from_heads, self_repair_from_layers):
    """Calculate the self-repair contribution from layer normalization (as residual)."""
    return self_repair - self_repair_from_heads - self_repair_from_layers


def get_self_repair_all(head, per_head_direct_effect, ablated_per_head_direct_effect, logit_diffs, direct_effects, 
        ablated_direct_effects, all_layer_direct_effect=None, ablated_all_layer_direct_effect=None):
    """
    Calculate all self-repair components if all required inputs are provided.
    Returns individual components based on available inputs.
    """
    # Calculate basic self-repair
    self_repair = calculate_self_repair(logit_diffs, direct_effects, ablated_direct_effects)
    
    # Calculate self-repair from heads
    self_repair_from_heads = calculate_self_repair_from_heads(head, per_head_direct_effect, ablated_per_head_direct_effect)
    
    # Calculate self-repair from layers if layer data is provided
    if all_layer_direct_effect is not None and ablated_all_layer_direct_effect is not None:
        self_repair_from_layers = calculate_self_repair_from_layers(all_layer_direct_effect, ablated_all_layer_direct_effect)
        
        # Calculate self-repair from layer norm as residual
        self_repair_from_LN = calculate_self_repair_from_LN(self_repair, self_repair_from_heads, self_repair_from_layers)
        
        return self_repair, self_repair_from_heads, self_repair_from_layers, self_repair_from_LN
    else:
        # Return only what we can calculate without layer data
        return self_repair, self_repair_from_heads, None, None



# Define tensor configurations for cleaner code
tensor_configs = {
    # Base measurements
    "logit_diffs": {"clipped": False},
    "direct_effects": {"clipped": False},
    "ablated_direct_effects": {"clipped": False},
    "self_repair_from_heads": {"clipped": False},
    "self_repair_from_layers": {"clipped": False},
    "self_repair_from_LN": {"clipped": False},
    
    # Percentage measurements (clipped)
    "percent_LN_of_DE": {
        "clipped": True, 
        "numerator": "self_repair_from_LN", 
        "denominator": "direct_effects"
    },
    "percent_heads_of_DE": {
        "clipped": True, 
        "numerator": "self_repair_from_heads", 
        "denominator": "direct_effects"
    },
    "percent_layers_of_DE": {
        "clipped": True, 
        "numerator": "self_repair_from_layers", 
        "denominator": "direct_effects"
    },
    "percent_self_repair_of_DE": {
        "clipped": True, 
        "numerator": "self_repair", 
        "denominator": "direct_effects"
    },
    
    # Percentage measurements (unclipped)
    "unclipped_percent_LN_of_DE": {
        "clipped": False, 
        "numerator": "self_repair_from_LN", 
        "denominator": "direct_effects"
    },
    "unclipped_percent_heads_of_DE": {
        "clipped": False, 
        "numerator": "self_repair_from_heads", 
        "denominator": "direct_effects"
    },
    "unclipped_percent_layers_of_DE": {
        "clipped": False, 
        "numerator": "self_repair_from_layers", 
        "denominator": "direct_effects"
    },
    "unclipped_percent_self_repair_of_DE": {
        "clipped": False, 
        "numerator": "self_repair", 
        "denominator": "direct_effects"
    },
}
# --- Configuration ---

@dataclass
class SelfRepairConfig:
    model_name: str = "pythia-160m"
    dataset_name: str = "pile"
    batch_size: int = 2
    prompt_len: int = 100
    min_tokens: int = 1_000
    percentile: float = 0.02  # For top instance selection
    metrics: List[str] = field(default_factory=lambda: [
        "logit_diffs", "direct_effects", "ablated_direct_effects", 
        "self_repair_from_heads", "self_repair_from_layers", "self_repair_from_LN", "self_repair",
        "percent_LN_of_DE", "percent_heads_of_DE", "percent_layers_of_DE", "percent_self_repair_of_DE",
        "unclipped_percent_LN_of_DE", "unclipped_percent_heads_of_DE", 
        "unclipped_percent_layers_of_DE", "unclipped_percent_self_repair_of_DE"
    ])
    save_results: bool = True
    save_folder: str = "data/pickle_storage/breakdown_self_repair/"
    device: torch.device = field(default_factory=lambda: torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # Internal calculated properties
    total_tokens: int = field(init=False)
    safe_model_name: str = field(init=False)
    num_batches: int = field(init=False)
    total_prompts: int = field(init=False)
    num_top_instances: int = field(init=False)
    percentile_str: str = field(init=False)

    def __post_init__(self):
        self.total_tokens = ((self.min_tokens // (self.prompt_len * self.batch_size)) + 1) * (self.prompt_len * self.batch_size)
        self.safe_model_name = self.model_name.replace("/", "_")
        # These will be set properly by the DataLoader
        self.num_batches = 0
        self.total_prompts = 0
        self.num_top_instances = 0
        self.percentile_str = "" if self.percentile == 0.02 else f"{self.percentile}_" # 0.02 is the default

        # Ensure save folder exists
        if self.save_results and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    @property
    def tensor_configs(self) -> Dict[str, Dict]:
        """Defines the structure and calculation method for each tensor."""
        all_configs = {
            # Base measurements
            "logit_diffs": {"clipped": False},
            "direct_effects": {"clipped": False},
            "ablated_direct_effects": {"clipped": False},
            "self_repair_from_heads": {"clipped": False},
            "self_repair_from_layers": {"clipped": False},
            "self_repair_from_LN": {"clipped": False},
            "self_repair": {"clipped": False}, # Added self_repair base measurement

            # Percentage measurements (clipped)
            "percent_LN_of_DE": {"clipped": True, "numerator": "self_repair_from_LN", "denominator": "direct_effects"},
            "percent_heads_of_DE": {"clipped": True, "numerator": "self_repair_from_heads", "denominator": "direct_effects"},
            "percent_layers_of_DE": {"clipped": True, "numerator": "self_repair_from_layers", "denominator": "direct_effects"},
            "percent_self_repair_of_DE": {"clipped": True, "numerator": "self_repair", "denominator": "direct_effects"},

            # Percentage measurements (unclipped)
            "unclipped_percent_LN_of_DE": {"clipped": False, "numerator": "self_repair_from_LN", "denominator": "direct_effects"},
            "unclipped_percent_heads_of_DE": {"clipped": False, "numerator": "self_repair_from_heads", "denominator": "direct_effects"},
            "unclipped_percent_layers_of_DE": {"clipped": False, "numerator": "self_repair_from_layers", "denominator": "direct_effects"},
            "unclipped_percent_self_repair_of_DE": {"clipped": False, "numerator": "self_repair", "denominator": "direct_effects"},
        }
        # Filter based on requested metrics
        return {name: config for name, config in all_configs.items() if name in self.metrics}


# --- Data Loading ---

class SelfRepairDataLoader:
    def __init__(self, model: HookedTransformer, config: SelfRepairConfig):
        self.model = model
        self.config = config
        self.dataset_iterator, self.config.num_batches = prepare_dataset(
            model=self.model,
            device=self.config.device,
            total_tokens_in_data=self.config.total_tokens,
            batch_size=self.config.batch_size,
            prompt_len=self.config.prompt_len,
            # return_generator=True, # Use generator
            dataset_name=self.config.dataset_name,
            padding=False
        )
        self.config.total_prompts = self.config.num_batches * self.config.batch_size
        self.config.num_top_instances = int(
            self.config.percentile * self.config.total_prompts * (self.config.prompt_len - 1)
        )

    def __iter__(self):
        return self.dataset_iterator

    def __len__(self):
        return self.config.num_batches


# --- Calculation ---

class SelfRepairCalculator:
    def __init__(self, model: HookedTransformer, config: SelfRepairConfig):
        self.model = model
        self.config = config
        self.tensor_configs = config.tensor_configs
        
        # Initialize result storage
        self.raw_tensors: Dict[str, Tensor] = {}
        self.condensed_tensors: Dict[str, Tensor] = {}
        self.full_tensors: Dict[str, Tensor] = {}

        self._initialize_tensors()

    def _initialize_tensors(self):
        """Initializes tensors based on config."""
        raw_tensor_shape = (
            self.config.total_prompts, 
            self.config.prompt_len - 1, 
            self.model.cfg.n_layers, 
            self.model.cfg.n_heads
        )
        condensed_full_shape = (self.model.cfg.n_layers, self.model.cfg.n_heads)

        self.raw_tensors = {name: torch.zeros(raw_tensor_shape, device='cpu') # Store raw on CPU potentially
                           for name in self.tensor_configs}
                           
        # Condensed tensors only for non-unclipped metrics (as per original logic)
        self.condensed_tensors = {
            f"condensed_{name}": torch.zeros(condensed_full_shape, device=self.config.device)
            for name in self.tensor_configs if "unclipped" not in name
        }
        self.full_tensors = {
            f"full_{name}": torch.zeros(condensed_full_shape, device=self.config.device)
            for name in self.tensor_configs
        }

    def _return_item(self, item):
        """Helper function for act_patch."""
        return item

    def _ablate_head_and_get_breakdown(
        self,
        head: tuple,
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        per_head_direct_effect: Tensor,
        all_layer_direct_effect: Tensor,
        cache: ActivationCache,
        logits: Tensor,
    ) -> Tuple[Tensor, ...]:
        """Calculates breakdown metrics after ablating a specific head."""
        
        # Run ablation and get new cache/logit/DEs
        ablated_cache: ActivationCache = act_patch(
            self.model, clean_tokens, [Node("z", head[0], head[1])], 
            self._return_item, corrupted_tokens, apply_metric_to_cache=True
        ) # type: ignore
        
        ablated_logits = act_patch(
            self.model, clean_tokens, [Node("z", head[0], head[1])],
            self._return_item, corrupted_tokens, apply_metric_to_cache=False,
        )
        
        ablated_per_head_direct_effect, ablated_all_layer_direct_effect = (
            collect_direct_effect(
                ablated_cache,
                correct_tokens=clean_tokens,
                model=self.model,
                collect_individual_neurons=False,
                cache_for_scaling=cache, # Use original cache for scaling
            )
        )

        # get the logit difference between everything
        correct_logits_score = get_correct_logit_score(logits, clean_tokens)
        ablated_logits_score = get_correct_logit_score(ablated_logits, clean_tokens)
        logit_diffs = ablated_logits_score - correct_logits_score

        # Get Direct Effect of Heads Pre/Post-Ablation
        direct_effects = per_head_direct_effect[head[0], head[1]]
        ablated_direct_effects = ablated_per_head_direct_effect[head[0], head[1]]

        # Calculate self-repair values
        self_repair = logit_diffs - (ablated_direct_effects - direct_effects)
        self_repair_from_heads = (
            ablated_per_head_direct_effect - per_head_direct_effect
        ).sum((0, 1)) - (ablated_per_head_direct_effect - per_head_direct_effect)[
            head[0], head[1]
        ]
        self_repair_from_layers = (
            ablated_all_layer_direct_effect - all_layer_direct_effect
        ).sum(0)
        # "self repair from LN" is the residual of total self-repair minus head/layer contributions
        self_repair_from_LN = self_repair - self_repair_from_heads - self_repair_from_layers

        return (
            logit_diffs, direct_effects, ablated_direct_effects,
            self_repair_from_heads, self_repair_from_layers, self_repair_from_LN,
            self_repair,
        )

    def compute_raw_metrics(self, data_loader: SelfRepairDataLoader):
        """Computes the raw, unaggregated metrics for all batches and heads."""
        print("Computing raw metrics...")
        pbar = tqdm(total=len(data_loader), desc="Processing batches")

        for batch_idx, clean_tokens, corrupted_tokens in data_loader:
            assert clean_tokens.shape == corrupted_tokens.shape == (self.config.batch_size, self.config.prompt_len)
            start_prompt_idx = batch_idx * self.config.batch_size
            end_prompt_idx = start_prompt_idx + self.config.batch_size

            # Ensure tokens are on the correct device
            clean_tokens = clean_tokens.to(self.config.device)
            corrupted_tokens = corrupted_tokens.to(self.config.device)

            # Cache clean model activations + direct effects
            logits, cache = self.model.run_with_cache(clean_tokens)
            per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(
                cache,
                correct_tokens=clean_tokens,
                model=self.model,
                collect_individual_neurons=False, # Keep False as per original
                cache_for_scaling=None
            )

            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    # Calculate metrics for this head ablation
                    results_tuple = self._ablate_head_and_get_breakdown(
                        (layer, head),
                        clean_tokens,
                        corrupted_tokens,
                        per_head_direct_effect,
                        all_layer_direct_effect,
                        cache,
                        logits,
                    )
                    
                    # Store results locally for percentage calculation
                    local_results = {
                        "logit_diffs": results_tuple[0],
                        "direct_effects": results_tuple[1],
                        "ablated_direct_effects": results_tuple[2],
                        "self_repair_from_heads": results_tuple[3],
                        "self_repair_from_layers": results_tuple[4],
                        "self_repair_from_LN": results_tuple[5],
                        "self_repair": results_tuple[6],
                    }

                    # Store base measurements
                    for name in local_results:
                        if name in self.raw_tensors:
                            self.raw_tensors[name][start_prompt_idx:end_prompt_idx, :, layer, head] = local_results[name].cpu()
                    
                    # Calculate and store percentages
                    for name, config in self.tensor_configs.items():
                        if "numerator" in config:
                            numerator_key = config["numerator"]
                            denominator_key = config["denominator"]
                            
                            # Ensure keys exist before division
                            if numerator_key in local_results and denominator_key in local_results:
                                numerator_val = local_results[numerator_key]
                                denominator_val = local_results[denominator_key]
                                
                                # Avoid division by zero or near-zero
                                denominator_val = torch.where(torch.abs(denominator_val) < 1e-6, torch.tensor(1e-6, device=denominator_val.device), denominator_val)

                                result = (numerator_val / denominator_val).cpu()
                                
                                if config["clipped"]:
                                    # Ensure result is float before clipping with numpy
                                    result = np.clip(result.float().numpy(), 0, 1)
                                    result = torch.from_numpy(result) # Convert back to tensor
                                    
                                self.raw_tensors[name][start_prompt_idx:end_prompt_idx, :, layer, head] = result
                            else:
                                print(f"Warning: Skipping metric {name} due to missing components ({numerator_key} or {denominator_key}).")


            pbar.update(1)
            # Optional: Clear CUDA cache periodically if memory is an issue
            if self.config.device.type == 'cuda':
                 torch.cuda.empty_cache()
                 
        pbar.close()
        print("Finished computing raw metrics.")

    def compute_aggregated_metrics(self):
        """Computes the condensed (top percentile) and full (average) metrics."""
        if not self.raw_tensors or "direct_effects" not in self.raw_tensors:
             print("Raw tensors not computed or missing 'direct_effects'. Skipping aggregation.")
             return
             
        print("Computing aggregated metrics...")
        
        # --- Process condensed tensors (top instances) ---
        if self.config.num_top_instances > 0:
            print(f"Selecting top {self.config.percentile*100:.2f}% instances ({self.config.num_top_instances})...")
            # Ensure direct_effects tensor is on the device for topk
            direct_effects_all = self.raw_tensors["direct_effects"].to(self.config.device)
            
            for layer in tqdm(range(self.model.cfg.n_layers), desc="Aggregating Condensed"):
                for head in range(self.model.cfg.n_heads):
                    # Get top indices based on direct effects for this head
                    top_indices = topk_of_Nd_tensor(
                        direct_effects_all[..., layer, head], self.config.num_top_instances
                    )

                    # Calculate condensed tensors
                    for name, raw_tensor in self.raw_tensors.items():
                        condensed_name = f"condensed_{name}"
                        if condensed_name in self.condensed_tensors:
                            # Move relevant slice to device for aggregation
                            data_for_top_indices = torch.stack([
                                raw_tensor[batch, pos, layer, head]
                                for batch, pos in top_indices
                            ]).to(self.config.device).float() # Ensure float for nanmean/mean
                            
                            if "percent" in name:
                                self.condensed_tensors[condensed_name][layer, head] = torch.nanmean(data_for_top_indices)
                            else:
                                self.condensed_tensors[condensed_name][layer, head] = torch.mean(data_for_top_indices)
            del direct_effects_all # Free memory
            if self.config.device.type == 'cuda': torch.cuda.empty_cache()
        else:
            print("Skipping condensed calculation as num_top_instances is 0.")

        # --- Calculate full tensors (averages over all instances) ---
        print("Aggregating Full...")
        for name, raw_tensor in tqdm(self.raw_tensors.items(), desc="Aggregating Full"):
            full_name = f"full_{name}"
            if full_name in self.full_tensors:
                 # Move tensor to device for aggregation, ensure float
                raw_tensor_device = raw_tensor.to(self.config.device).float()
                if "percent" in name:
                    self.full_tensors[full_name] = torch.nanmean(raw_tensor_device, dim=(0, 1))
                else:
                    self.full_tensors[full_name] = torch.mean(raw_tensor_device, dim=(0, 1))
        
        if self.config.device.type == 'cuda': torch.cuda.empty_cache()
        print("Finished computing aggregated metrics.")


    def get_results(self) -> Dict[str, Tensor]:
        """Returns all computed tensors."""
        return {**self.condensed_tensors, **self.full_tensors}


# --- Pipeline ---

class SelfRepairPipeline:
    def __init__(self, config: SelfRepairConfig):
        self.config = config
        print(f"Using device: {self.config.device}")
        print("Initializing model...")
        self.model = self._load_model()
        print("Initializing data loader...")
        self.data_loader = SelfRepairDataLoader(self.model, self.config)
        print("Initializing calculator...")
        self.calculator = SelfRepairCalculator(self.model, self.config)
        self.results: Dict[str, Tensor] = {}

    def _load_model(self) -> HookedTransformer:
        """Loads the HuggingFace model."""
        model = HookedTransformer.from_pretrained(
            self.config.model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=False, # Keep False as per original
            device=self.config.device,
        )
        model.set_use_attn_result(True)  # Needed for direct effect calculation
        return model

    def run(self):
        """Runs the full self-repair computation pipeline."""
        self.calculator.compute_raw_metrics(self.data_loader)
        self.calculator.compute_aggregated_metrics()
        self.results = self.calculator.get_results()

        if self.config.save_results:
            self.save_results()

        print("Pipeline finished.")
        return self.results

    def save_results(self):
        """Saves the computed tensors to pickle files."""
        if not self.results:
            print("No results to save.")
            return

        print(f"Saving results to {self.config.save_folder}...")
        for tensor_name, tensor_data in self.results.items():
             # Ensure tensor is on CPU before saving
            tensor_data_cpu = tensor_data.cpu()
            file_path = os.path.join(
                 self.config.save_folder, 
                 f"{self.config.percentile_str}{self.config.safe_model_name}_{tensor_name}.pkl"
            )
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(tensor_data_cpu, f)
            except Exception as e:
                 print(f"Error saving tensor {tensor_name} to {file_path}: {e}")
        print("Finished saving results.")


# --- Example Usage ---

if __name__ == "__main__":
    # 1. Configure the pipeline
    config = SelfRepairConfig(
        model_name="pythia-160m", 
        dataset_name="pile",
        batch_size=4, # Adjust batch size based on GPU memory
        min_tokens=500, # Smaller run for example
        percentile=0.05,
        save_results=True,
        save_folder="data/output/self_repair_results/"
        # Add/modify other configurations as needed
    )

    # 2. Create and run the pipeline
    pipeline = SelfRepairPipeline(config)
    results = pipeline.run()

    # 3. Access results (optional)
    # print("Computed results keys:", results.keys())
    # if "condensed_direct_effects" in results:
    #     print("Shape of condensed_direct_effects:", results["condensed_direct_effects"].shape)
