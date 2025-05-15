import torch
from torch import Tensor
from tqdm import tqdm
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional, Literal, Union
from functools import partial

from tf_perturb.direct_effect import (
    collect_direct_effect, # Still needed for instance selection
    get_correct_logit_score,
    topk_of_Nd_tensor,
)
import transformer_lens as ts
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint

from tf_perturb.dataset import prepare_dataset
from tf_perturb.path_patching import Node, act_patch # act_patch needed for ablation


# --- Helper Functions ---

def hook_fn_patch_or_freeze(
    activation: Tensor, 
    hook: HookPoint, 
    source_cache: ActivationCache, 
    target_node: Optional[Node] = None, # Node to ablate
    ablation_value: Optional[Tensor] = None, # Value to patch in for ablation (e.g., from corrupted run or zeros)
    freeze_mask: Optional[torch.Tensor] = None # Boolean mask for freezing downstream layers
) -> Tensor:
    """
    General hook function for patching/freezing.
    - If hook name matches target_node, patches with ablation_value.
    - If hook name is downstream (freeze_mask is True for this layer), patches with source_cache.
    - Otherwise, returns original activation.
    """
    is_target_ablation_hook = target_node is not None and hook.name == target_node.activation_name
    is_downstream_freeze_hook = freeze_mask is not None and hook.layer() is not None and hook.layer() < len(freeze_mask) and freeze_mask[hook.layer()]

    if is_target_ablation_hook:
        # Apply ablation patch (e.g., zeroing or resampling)
        # Assuming simple head ablation ('z') for now, matching original act_patch usage
        if ablation_value is not None and target_node.head is not None:
            # Simple head zeroing for now, adapt if resampling needed
            activation[:, :, target_node.head, :] = 0.0 
            # Or potentially use ablation_value if it's pre-calculated for the specific head
            # activation[:, :, target_node.head, :] = ablation_value[:, :, target_node.head, :]
        else:
            # Fallback or handle non-head cases if necessary
             pass # No change if ablation value/target doesn't fit expected pattern
        return activation
        
    elif is_downstream_freeze_hook:
        # Freeze by patching from the clean cache
        # Ensure dimensions match, might need adjustment based on specific hook point shapes
        try:
            cached_act = source_cache[hook.name]
            if activation.shape == cached_act.shape:
                 activation[:] = cached_act[:]
            # else:
            #     print(f"Warning: Shape mismatch freezing {hook.name}. Activation: {activation.shape}, Cache: {cached_act.shape}")
        except KeyError:
             print(f"Warning: Hook {hook.name} not found in source_cache for freezing.")
        return activation
        
    else:
        # No modification
        return activation

# --- Configuration ---

@dataclass
class SelfRepairConfigNew:
    model_name: str = "pythia-160m"
    dataset_name: str = "pile"
    batch_size: int = 2
    prompt_len: int = 100
    min_tokens: int = 1_000
    percentile: float = 0.02  # For top instance selection based on DE
    metrics: List[str] = field(default_factory=lambda: [
        "direct_effects", # Keep for instance selection
        "logit_clean_score", 
        "logit_ablated_score", 
        "logit_iso_ablated_score", 
        "self_repair_new"
    ])
    save_results: bool = True
    save_folder: str = "data/pickle_storage/new_self_repair/"
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
        self.percentile_str = "" if self.percentile == 0.02 else f"{self.percentile}_"  # 0.02 is the default

        # Ensure save folder exists
        if self.save_results and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    @property
    def tensor_configs(self) -> Dict[str, Dict]:
        """Defines the structure for the new metrics."""
        # For the new method, we just store the calculated scores directly.
        # No complex numerator/denominator needed for the main metric.
        all_configs = {
            "direct_effects": {}, # Still needed for top-k selection
            "logit_clean_score": {},
            "logit_ablated_score": {},
            "logit_iso_ablated_score": {},
            "self_repair_new": {},
        }
        # Filter based on requested metrics
        return {name: config for name, config in all_configs.items() if name in self.metrics}


# --- Data Loading ---

class SelfRepairDataLoaderNew:
    def __init__(self, model: HookedTransformer, config: SelfRepairConfigNew):
        self.model = model
        self.config = config
        # Assuming prepare_dataset returns tuples: (batch_idx, clean_tokens, corrupted_tokens)
        self.dataset_iterator, self.config.num_batches = prepare_dataset(
            model=self.model,
            device=self.config.device, # prepare_dataset might handle device internally
            n_tokens=self.config.total_tokens,
            batch_size=self.config.batch_size,
            prompt_len=self.config.prompt_len,
            return_generator=True, # Use generator
            dataset_name=self.config.dataset_name
        )
        self.config.total_prompts = self.config.num_batches * self.config.batch_size
        self.config.num_top_instances = int(
            self.config.percentile * self.config.total_prompts * (self.config.prompt_len - 1)
        )
        print(f"DataLoader initialized: {self.config.num_batches} batches, {self.config.total_prompts} total prompts.")


    def __iter__(self):
        # Ensure tokens are moved to the correct device if not handled by prepare_dataset
        for batch_data in self.dataset_iterator:
            yield tuple(d.to(self.config.device) if isinstance(d, torch.Tensor) else d for d in batch_data)

    def __len__(self):
        return self.config.num_batches


# --- Calculation ---

class SelfRepairCalculatorNew:
    def __init__(self, model: HookedTransformer, config: SelfRepairConfigNew):
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
        # Only store raw tensors per instance (prompt, pos, layer, head)
        # Aggregated tensors (condensed/full) will have shape (layer, head)
        condensed_full_shape = (self.model.cfg.n_layers, self.model.cfg.n_heads)

        # Initialize raw tensors on CPU to save GPU RAM during computation phase
        self.raw_tensors = {name: torch.zeros(raw_tensor_shape, device='cpu') 
                           for name in self.tensor_configs}
                           
        # Initialize aggregation tensors on the target device
        self.condensed_tensors = {
            f"condensed_{name}": torch.zeros(condensed_full_shape, device=self.config.device)
            for name in self.tensor_configs # Aggregate all metrics
        }
        self.full_tensors = {
            f"full_{name}": torch.zeros(condensed_full_shape, device=self.config.device)
            for name in self.tensor_configs # Aggregate all metrics
        }
        print(f"Initialized raw tensors: {self.raw_tensors.keys()}")


    def _return_item(self, item):
        """Helper function for act_patch."""
        return item

    def _run_ablation(
        self, 
        head_node: Node, 
        clean_tokens: Tensor, 
        corrupted_tokens: Tensor # Used to get the value to patch in (resampling)
    ) -> Tensor:
        """Runs the model with the specified head ablated (using resampling)."""
        # Use act_patch for consistency with original method's ablation style (resampling)
        # Note: apply_metric_to_cache=False returns logits
        ablated_logits = act_patch(
            self.model, 
            clean_tokens, 
            [head_node], # Patch this node
            self._return_item, # Patching function (just return the item for act_patch's internal logic)
            corrupted_tokens, # Source of new activations for the patch
            apply_metric_to_cache=False,
        )
        ablated_score = get_correct_logit_score(ablated_logits, clean_tokens)
        return ablated_score

    def _run_iso_ablation(
        self, 
        head_node: Node, 
        clean_tokens: Tensor, 
        clean_cache: ActivationCache,
        corrupted_tokens: Tensor # Used for the ablation value itself
    ) -> Tensor:
        """Runs the model with head ablated AND downstream layers frozen."""
        
        ablation_layer = head_node.layer
        hooks = []

        # 1. Hook to ablate the target head 'z' output
        # We need the *value* from the corrupted run for this head to patch in
        # This requires a separate run or careful caching if not readily available
        # For simplicity matching `act_patch`, let's just zero it out for now.
        # A more accurate implementation would fetch the specific head's 'z' output
        # from a corrupted run cache.
        def ablate_hook_fn(activation: Tensor, hook: HookPoint):
            if hook.name == head_node.activation_name:
                 activation[:, :, head_node.head, :] = 0.0 # Zero out the head
            return activation
        hooks.append((head_node.activation_name, ablate_hook_fn))


        # 2. Hooks to freeze downstream layers
        freeze_points = []
        for layer_idx in range(ablation_layer + 1, self.model.cfg.n_layers):
             # Freeze the *inputs* to the attention and MLP blocks of subsequent layers
             # This prevents them from reacting to the ablated signal propagating residually.
             freeze_points.extend([
                 f"blocks.{layer_idx}.ln1.hook_normalized", # Input to Attn
                 f"blocks.{layer_idx}.ln2.hook_normalized", # Input to MLP
                 # Potentially freeze resid_mid as well if needed? Test with inputs first.
                 # f"blocks.{layer_idx}.hook_resid_mid" 
             ])
             
        # Freeze final LN and unembed? Usually not needed if freezing block inputs.
        # freeze_points.append("ln_final.hook_normalized")

        def freeze_hook_fn(activation: Tensor, hook: HookPoint):
            # Patch from clean cache
            try:
                cached_act = clean_cache[hook.name]
                if activation.shape == cached_act.shape:
                    activation[:] = cached_act[:]
                # else: Handle shape mismatch if necessary 
            except KeyError:
                 print(f"Warning: Hook {hook.name} not in clean_cache for freezing.")
            return activation

        for point in freeze_points:
             if point in self.model.hook_dict: # Only add hooks that exist
                 hooks.append((point, freeze_hook_fn))
             # else: print(f"Skipping non-existent freeze hook: {point}")


        # Run with hooks
        iso_ablated_logits = self.model.run_with_hooks(
            clean_tokens,
            fwd_hooks=hooks,
            return_type="logits"
        )
        
        iso_ablated_score = get_correct_logit_score(iso_ablated_logits, clean_tokens)
        return iso_ablated_score


    def _compute_new_metrics_for_head(
        self,
        head: Tuple[int, int],
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        clean_cache: ActivationCache,
        clean_logit_score: Tensor,
    ) -> Dict[str, Tensor]:
        """Computes the new self-repair and related logit scores for one head."""
        
        head_node = Node("z", layer=head[0], head=head[1])

        # 1. Run full ablation (downstream adapts)
        logit_ablated_score = self._run_ablation(head_node, clean_tokens, corrupted_tokens)

        # 2. Run iso-ablation (downstream frozen)
        logit_iso_ablated_score = self._run_iso_ablation(head_node, clean_tokens, clean_cache, corrupted_tokens)

        # 3. Calculate new self-repair metric
        self_repair_new = logit_ablated_score - logit_iso_ablated_score

        return {
            "logit_clean_score": clean_logit_score,
            "logit_ablated_score": logit_ablated_score,
            "logit_iso_ablated_score": logit_iso_ablated_score,
            "self_repair_new": self_repair_new,
        }

    def compute_raw_metrics(self, data_loader: SelfRepairDataLoaderNew):
        """Computes the raw, unaggregated metrics for all batches and heads using the new method."""
        print("Computing raw metrics (New Method)...")
        pbar = tqdm(total=len(data_loader), desc="Processing batches")

        for batch_idx, clean_tokens, corrupted_tokens in data_loader:
            # Tokens should already be on device from DataLoaderNew.__iter__
            assert clean_tokens.shape[0] == self.config.batch_size
            start_prompt_idx = batch_idx * self.config.batch_size
            end_prompt_idx = start_prompt_idx + self.config.batch_size

            # --- Precompute Clean Run results ---
            logits_clean, clean_cache = self.model.run_with_cache(clean_tokens)
            clean_logit_score_batch = get_correct_logit_score(logits_clean, clean_tokens) # Shape: (batch, seq-1)
            
            # Precompute Direct Effects *only if needed for instance selection*
            if "direct_effects" in self.raw_tensors:
                 per_head_direct_effect, _ = collect_direct_effect(
                     clean_cache,
                     correct_tokens=clean_tokens,
                     model=self.model,
                     display=False,
                     collect_individual_neurons=False, 
                 )
            
            # --- Iterate through heads ---
            for layer in range(self.model.cfg.n_layers):
                for head in range(self.model.cfg.n_heads):
                    
                    # Compute new metrics
                    new_metrics = self._compute_new_metrics_for_head(
                        (layer, head),
                        clean_tokens,
                        corrupted_tokens,
                        clean_cache,
                        clean_logit_score_batch, # Pass the precomputed batch scores
                    )

                    # Store results in raw_tensors (move to CPU)
                    target_slice = (slice(start_prompt_idx, end_prompt_idx), slice(None), layer, head)
                    
                    for name, value in new_metrics.items():
                        if name in self.raw_tensors:
                            self.raw_tensors[name][target_slice] = value.cpu()

                    # Store direct effect if computed
                    if "direct_effects" in self.raw_tensors:
                         de_value = per_head_direct_effect[layer, head] # Shape: (batch, seq-1)
                         self.raw_tensors["direct_effects"][target_slice] = de_value.cpu()


            pbar.update(1)
            # Optional: Clear CUDA cache 
            if self.config.device.type == 'cuda':
                 torch.cuda.empty_cache()
                 
        pbar.close()
        print("Finished computing raw metrics (New Method).")

    def compute_aggregated_metrics(self):
        """Computes condensed (top DE percentile) and full (average) metrics."""
        if not self.raw_tensors:
             print("Raw tensors not computed. Skipping aggregation.")
             return
             
        print("Computing aggregated metrics (New Method)...")
        
        # --- Process condensed tensors (top DE instances) ---
        if self.config.num_top_instances > 0 and "direct_effects" in self.raw_tensors:
            print(f"Selecting top {self.config.percentile*100:.2f}% DE instances ({self.config.num_top_instances})...")
            # Move direct_effects to device for topk calculation
            direct_effects_all = self.raw_tensors["direct_effects"].to(self.config.device)
            
            for layer in tqdm(range(self.model.cfg.n_layers), desc="Aggregating Condensed"):
                for head in range(self.model.cfg.n_heads):
                    # Get top indices based on direct effects
                    top_indices = topk_of_Nd_tensor(
                        direct_effects_all[..., layer, head], self.config.num_top_instances
                    )

                    # Calculate condensed tensors by averaging *new* metrics over these top DE indices
                    for name, raw_tensor in self.raw_tensors.items():
                        condensed_name = f"condensed_{name}"
                        if condensed_name in self.condensed_tensors:
                            # Move relevant slice to device for aggregation
                            data_for_top_indices = torch.stack([
                                raw_tensor[batch, pos, layer, head]
                                for batch, pos in top_indices # top_indices are (batch_idx, pos_idx) tuples
                            ]).to(self.config.device).float() # Ensure float for nanmean/mean
                            
                            # Use nanmean to handle potential NaNs, mean otherwise
                            if torch.isnan(data_for_top_indices).any():
                                self.condensed_tensors[condensed_name][layer, head] = torch.nanmean(data_for_top_indices)
                            else:
                                self.condensed_tensors[condensed_name][layer, head] = torch.mean(data_for_top_indices)
                                
            del direct_effects_all # Free memory
            if self.config.device.type == 'cuda': torch.cuda.empty_cache()
        elif self.config.num_top_instances == 0:
             print("Skipping condensed calculation as num_top_instances is 0.")
        else:
             print("Skipping condensed calculation as 'direct_effects' are not being computed/stored.")


        # --- Calculate full tensors (averages over all instances) ---
        print("Aggregating Full...")
        for name, raw_tensor in tqdm(self.raw_tensors.items(), desc="Aggregating Full"):
            full_name = f"full_{name}"
            if full_name in self.full_tensors:
                 # Move tensor to device for aggregation, ensure float
                raw_tensor_device = raw_tensor.to(self.config.device).float()
                
                # Calculate mean over batch and position dimensions
                if torch.isnan(raw_tensor_device).any():
                     self.full_tensors[full_name] = torch.nanmean(raw_tensor_device, dim=(0, 1))
                else:
                     self.full_tensors[full_name] = torch.mean(raw_tensor_device, dim=(0, 1))
        
        if self.config.device.type == 'cuda': torch.cuda.empty_cache()
        print("Finished computing aggregated metrics (New Method).")


    def get_results(self) -> Dict[str, Tensor]:
        """Returns all computed aggregated tensors."""
        # Combine condensed and full tensors into one dictionary
        # Filter out empty tensors if condensed wasn't calculated
        results = {**self.full_tensors}
        if self.config.num_top_instances > 0:
            results.update(self.condensed_tensors)
            
        # Make sure tensors exist before returning
        return {k: v for k, v in results.items() if v.numel() > 0}


# --- Pipeline ---

class SelfRepairPipelineNew:
    def __init__(self, config: SelfRepairConfigNew):
        self.config = config
        print(f"Using device: {self.config.device}")
        print("Initializing model...")
        self.model = self._load_model()
        print("Initializing data loader...")
        self.data_loader = SelfRepairDataLoaderNew(self.model, self.config)
        print("Initializing calculator...")
        self.calculator = SelfRepairCalculatorNew(self.model, self.config)
        self.results: Dict[str, Tensor] = {}

    def _load_model(self) -> HookedTransformer:
        """Loads the HuggingFace model."""
        model = HookedTransformer.from_pretrained(
            self.config.model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=False, 
            device=self.config.device,
        )
        # Needed for ActivationCache and potentially some hook points
        model.set_use_attn_result(True) 
        model.set_use_split_qkv_input(True) # Might be needed for specific freeze points if used
        model.set_use_hook_mlp_in(True) # Might be needed for specific freeze points if used
        return model

    def run(self) -> Dict[str, Tensor]:
        """Runs the full self-repair computation pipeline."""
        self.calculator.compute_raw_metrics(self.data_loader)
        self.calculator.compute_aggregated_metrics()
        self.results = self.calculator.get_results()

        if self.config.save_results:
            self.save_results()

        print("Pipeline finished.")
        return self.results

    def save_results(self):
        """Saves the computed aggregated tensors to pickle files."""
        if not self.results:
            print("No results to save.")
            return

        print(f"Saving results to {self.config.save_folder}...")
        save_dir = self.config.save_folder
        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists

        for tensor_name, tensor_data in self.results.items():
             # Ensure tensor is on CPU before saving
            tensor_data_cpu = tensor_data.cpu()
            file_path = os.path.join(
                 save_dir, 
                 f"{self.config.percentile_str}{self.config.safe_model_name}_{tensor_name}.pkl"
            )
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(tensor_data_cpu, f)
            except Exception as e:
                 print(f"Error saving tensor {tensor_name} to {file_path}: {e}")
        print(f"Finished saving results to {save_dir}")


# --- Example Usage ---

if __name__ == "__main__":
    # 1. Configure the pipeline
    config = SelfRepairConfigNew(
        model_name="pythia-160m", #"gpt2-small", # 
        dataset_name="pile",
        batch_size=2, # Adjust batch size based on GPU memory
        min_tokens=200, # Smaller run for quick example
        percentile=0.05, # Use 5% DE for selecting instances for condensed analysis
        save_results=True,
        save_folder="data/output/new_self_repair_results/",
        metrics=[ # Specify which metrics to compute and store
            "direct_effects", # Needed for condensed aggregation
            "logit_clean_score", 
            "logit_ablated_score", 
            "logit_iso_ablated_score", 
            "self_repair_new"
        ]
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Explicitly set CUDA if preferred
    )

    # 2. Create and run the pipeline
    pipeline = SelfRepairPipelineNew(config)
    results = pipeline.run()

    # 3. Access results (optional)
    print("\nComputed results keys:", list(results.keys()))
    if "condensed_self_repair_new" in results:
        print("Shape of condensed_self_repair_new:", results["condensed_self_repair_new"].shape)
        print("Example value (L0H0):", results["condensed_self_repair_new"][0, 0].item())
    if "full_self_repair_new" in results:
        print("Shape of full_self_repair_new:", results["full_self_repair_new"].shape)
        print("Example value (L0H0):", results["full_self_repair_new"][0, 0].item())

