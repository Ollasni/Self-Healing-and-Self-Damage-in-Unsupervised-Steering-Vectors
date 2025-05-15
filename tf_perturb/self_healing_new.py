import torch
from torch import Tensor
from tqdm import tqdm
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal, Callable
from functools import partial

from tf_perturb.direct_effect import (
    collect_direct_effect,  # Still needed for instance selection
    get_correct_logit_score,
    topk_of_Nd_tensor,
)
from transformer_lens import HookedTransformer, ActivationCache, utils
from transformer_lens.hook_points import HookPoint

from tf_perturb.dataset import prepare_dataset


# --- Intervention hook generator ---
def make_head_intervention_hook(
    layer: int,
    head: int,
    steering: Tensor,
    mode: Literal['add', 'replace'],
    apply_ln: bool,
    model: HookedTransformer
) -> Callable:
    """
    Returns a hook function that either adds or replaces the output of a specific head.
    steering: [batch, seq, d_model]
    mode: 'add' or 'replace'
    apply_ln: whether to apply final LayerNorm to the intervention
    """
    hook_name = utils.get_act_name('z', layer)
    # Precompute LN operator if needed
    ln = None
    if apply_ln:
        ln = model.cfg
    def hook(z: Tensor, hook: HookPoint) -> Tensor:
        # z: [batch, seq, n_head, d_model]
        # extract intervention for this head
        iv = steering[..., None, :].clone()
        # optionally normalize iv via final ln
        if apply_ln:
            iv = model.apply_ln(iv, layer=-1)
        if mode == 'add':
            z[:, :, head, :] = z[:, :, head, :] + iv.squeeze(2)
        else:  # 'replace'
            z[:, :, head, :] = iv.squeeze(2)
        return z
    return hook, hook_name

# Define the patching hook function similar to the notebook
def patch_head_z(
    z: Tensor,
    hook: HookPoint,
    corrupted_cache: ActivationCache,
    head_index: int,
    position_indices: Optional[slice] = None,  # Add position slicing if needed
) -> Tensor:
    """Patches the output of a given head (dim 2) with values from corrupted_cache."""
    try:
        new_z = corrupted_cache[hook.name]
        # Ensure shapes match for patching
        if z.shape != new_z.shape:
            print(
                f"Warning: Shape mismatch patching {hook.name}. "
                f"Clean: {z.shape}, Corrupt: {new_z.shape}"
            )
            return z  # Return original if shapes don't match

        if position_indices is not None:
            z[:, position_indices, head_index, :] = new_z[
                :, position_indices, head_index, :
            ]
        else:
            z[:, :, head_index, :] = new_z[:, :, head_index, :]
        return z
    except KeyError:
        print(f"Warning: Hook {hook.name} not found in corrupted_cache for patching.")
        return z
    except Exception as e:
        print(f"Error during patch_head_z for {hook.name}: {e}")
        return z


# Define the freezing hook function
def freeze_hook_fn(activation: Tensor, hook: HookPoint, clean_cache: ActivationCache):
    """Freezes activations by patching from the clean_cache."""
    try:
        cached_act = clean_cache[hook.name]
        if activation.shape == cached_act.shape:
            # Use clone().detach() to avoid modifying cache and ensure no grad issues
            activation[:] = cached_act.clone().detach()
        else:
            print(
                f"Warning: Shape mismatch freezing {hook.name}. "
                f"Act: {activation.shape}, Cache: {cached_act.shape}"
            )
    except KeyError:
        print(f"Warning: Hook {hook.name} not in clean_cache for freezing.")
    except Exception as e:
        print(f"Error during freeze_hook_fn for {hook.name}: {e}")
    return activation


@dataclass
class SelfRepairConfigNew:
    model_name: str = "pythia-160m"
    dataset_name: str = "pile"
    batch_size: int = 2
    prompt_len: int = 100
    total_tokens_in_data: int = 1_000
    percentile: float = 0.02  # For top instance selection based on DE
    metrics: List[str] = field(
        default_factory=lambda: [
            "direct_effects",  # Keep for instance selection
            "logit_clean_score",
            "logit_ablated_score",
            "logit_iso_ablated_score",
            "self_repair_new",
        ]
    )
    save_results: bool = True
    save_folder: str = "data/pickle_storage/new_self_repair/"
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "mps" if torch.backends.mps.is_available() else 
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Internal calculated properties
    safe_model_name: str = field(init=False)
    num_batches: int = field(init=False)
    total_prompts: int = field(init=False)
    num_top_instances: int = field(init=False)
    percentile_str: str = field(init=False)

    def __post_init__(self):
        self.safe_model_name = self.model_name.replace("/", "_")
        # These will be set properly by the DataLoader
        self.num_batches = 0
        self.total_prompts = 0
        self.num_top_instances = 0
        self.percentile_str = (
            ""
            if self.percentile == 0.02
            else f"{self.percentile:.2f}_"  # Format percentile
        )

        # Ensure save folder exists
        if self.save_results and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    @property
    def tensor_configs(self) -> Dict[str, Dict]:
        """Defines the structure for the new metrics."""
        all_configs = {
            "direct_effects": {},
            "logit_clean_score": {},
            "logit_ablated_score": {},
            "logit_iso_ablated_score": {},
            "self_repair_new": {},
        }
        return {
            name: config for name, config in all_configs.items() if name in self.metrics
        }


class SelfRepairDataLoaderNew:
    def __init__(self, model: HookedTransformer, config: SelfRepairConfigNew):
        self.model = model
        self.config = config
        self.dataset_iterator, self.config.num_batches = prepare_dataset(
            model=self.model,
            device=self.config.device,
            total_tokens_in_data=self.config.total_tokens_in_data,
            batch_size=self.config.batch_size,
            prompt_len=self.config.prompt_len,
            dataset_name=self.config.dataset_name,
            padding=False,
        )
        self.config.total_prompts = self.config.num_batches * self.config.batch_size
        # Ensure prompt_len > 1 for valid calculation
        prompt_len_for_calc = max(1, self.config.prompt_len - 1)
        self.config.num_top_instances = int(
            self.config.percentile * self.config.total_prompts * prompt_len_for_calc
        )
        print(
            f"DataLoader initialized: {self.config.num_batches} batches, "
            f"{self.config.total_prompts} total prompts."
        )

    def __iter__(self):
        for batch_data in self.dataset_iterator:
            yield tuple(
                d.to(self.config.device) if isinstance(d, torch.Tensor) else d
                for d in batch_data
            )

    def __len__(self):
        return self.config.num_batches


# --- Calculation ---


class SelfRepairCalculatorNew:
    def __init__(self, model: HookedTransformer, config: SelfRepairConfigNew):
        self.model = model
        self.config = config
        self.tensor_configs = config.tensor_configs
        self.raw_tensors: Dict[str, Tensor] = {}
        self.condensed_tensors: Dict[str, Tensor] = {}
        self.full_tensors: Dict[str, Tensor] = {}
        self._initialize_tensors()

    def _initialize_tensors(self):
        """Initializes tensors based on config."""
        if self.config.prompt_len <= 1:
            print("Warning: prompt_len <= 1, raw tensor shape will be zero in seq dim.")
        raw_tensor_shape = (
            self.config.total_prompts,
            max(0, self.config.prompt_len - 1),  # Ensure non-negative seq dim
            self.model.cfg.n_layers,
            self.model.cfg.n_heads,
        )
        condensed_full_shape = (self.model.cfg.n_layers, self.model.cfg.n_heads)

        self.raw_tensors = {
            name: torch.zeros(raw_tensor_shape, device="cpu")
            for name in self.tensor_configs
        }
        self.condensed_tensors = {
            f"condensed_{name}": torch.zeros(
                condensed_full_shape, device=self.config.device
            )
            for name in self.tensor_configs
        }
        self.full_tensors = {
            f"full_{name}": torch.zeros(condensed_full_shape, device=self.config.device)
            for name in self.tensor_configs
        }
        print(f"Initialized raw tensors: {list(self.raw_tensors.keys())}")

    # --- single‐pass intervention (was _run_ablation) ---
    def _run_intervention(
        self,
        layer: int,
        head: int,
        steering: Tensor,               # your steering vector
        mode: Literal['add','replace'], # 'add' or 'replace'
        apply_ln: bool,                 # whether to LN your vector
        clean_tokens: Tensor,
    ) -> Tensor:
        """
        Inject steering into one head and let the rest of the model run normally.
        """
        hook_fn, hook_name = make_head_intervention_hook(
            layer, head, steering, mode, apply_ln, self.model
        )
        logits = self.model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits",
        )
        return get_correct_logit_score(logits, clean_tokens)

    # --- isolated intervention with downstream frozen (was _run_iso_ablation) ---
    def _run_iso_intervention(
        self,
        layer: int,
        head: int,
        steering: Tensor,
        mode: Literal['add','replace'],
        apply_ln: bool,
        clean_tokens: Tensor,
        clean_cache_for_freezing: ActivationCache,
    ) -> Tensor:
        """
        Inject steering into one head, then freeze ALL
        subsequent blocks to their clean‐run activations.
        """
        # 1) our head intervention hook
        hook_fn, hook_name = make_head_intervention_hook(
            layer, head, steering, mode, apply_ln, self.model
        )
        hooks = [(hook_name, hook_fn)]

        # 2) freeze every downstream normalized output
        for lyr in range(layer+1, self.model.cfg.n_layers):
            for sub in ("ln1", "ln2"):
                name = utils.get_act_name("normalized", lyr, sub)
                if name in self.model.hook_dict:
                    hooks.append((name, partial(freeze_hook_fn, clean_cache=clean_cache_for_freezing)))

        # run
        logits = self.model.run_with_hooks(
            clean_tokens,
            fwd_hooks=hooks,
            return_type="logits",
        )
        return get_correct_logit_score(logits, clean_tokens)
    
    def _get_clean_caches(self, tokens: Tensor) -> Tuple[Tensor, ActivationCache, ActivationCache]:
        """
        Run the model once on `tokens`, caching both:
        - every head’s raw post–attention output (`*.attn.hook_z`)
        - every normalized activation downstream (`*.hook_normalized`)
        Returns (logits, z_cache, norm_cache).
        """
        logits, full_cache = self.model.run_with_cache(
            tokens,
            return_type="logits",
            names_filter=lambda name: name.endswith("attn.hook_z")
                                or "hook_normalized" in name,
        )
        # split into two ActivationCaches
        z_cache   = ActivationCache({k: v for k, v in full_cache.items() if k.endswith("attn.hook_z")}, model=self.model)
        norm_cache= ActivationCache({k: v for k, v in full_cache.items() if "hook_normalized" in k}, model=self.model)
        return logits, z_cache, norm_cache

    def _compute_metrics_for_head(
        self,
        head_indices: Tuple[int,int],
        clean_tokens: Tensor,
        clean_cache_for_freezing: ActivationCache,
        steering: Tensor
    ) -> Dict[str,Tensor]:
        """Computes the ablation, iso-ablation, and self-repair scores for one head."""
        layer, head = head_indices

        # Steer is your precomputed [batch, seq, d_model] vector for this head
        # Or Run full ablation (downstream adapts)
        ablated_score = self._run_intervention(
            layer,
            head,
            steering=steering,
            mode="add",
            apply_ln=False,
            clean_tokens=clean_tokens,
        )

        # Run iso-ablation (downstream frozen)
        iso_score = self._run_iso_intervention(
            layer,
            head,
            steering=steering,
            mode="add",
            apply_ln=False,
            clean_tokens=clean_tokens,
            clean_cache_for_freezing=clean_cache_for_freezing,
        )

        # Calculate new self-repair metric
        self_repair = ablated_score - iso_score

        return {
            # Note: clean_logit_score is handled per-batch outside this func
            "logit_ablated_score": ablated_score,
            "logit_iso_ablated_score": iso_score,
            "self_repair_new": self_repair,
        }

    def compute_raw_metrics(self, data_loader: SelfRepairDataLoaderNew):
        if self.config.prompt_len <= 1:
            return

        pbar = tqdm(total=len(data_loader), desc="Processing batches")
        for batch_idx, clean_tokens, corrupted_tokens in data_loader:
            start = batch_idx * self.config.batch_size
            end = start + self.config.batch_size

            # --- single clean pass, split caches --- #
            logits_clean, z_cache, norm_cache = self._get_clean_caches(clean_tokens)
            clean_score = get_correct_logit_score(logits_clean, clean_tokens)

            # --- one pass on corrupted to get steering-only cache_z --- #
            _, corrupt_z_cache, _ = self._get_clean_caches(corrupted_tokens)

            # --- optional direct effects --- #
            per_head_de = None
            if "direct_effects" in self.raw_tensors:
                _, full_clean_cache = self.model.run_with_cache(clean_tokens)
                per_head_de, _ = collect_direct_effect(
                    full_clean_cache, correct_tokens=clean_tokens,
                    model=self.model, collect_individual_neurons=False,
                    cache_for_scaling=full_clean_cache
                )
                del full_clean_cache

            # --- loop over heads --- #
            for layer in range(self.model.cfg.n_layers):
                hook_name = utils.get_act_name("z", layer)
                z_clean   = z_cache[hook_name]
                z_corrupt = corrupt_z_cache[hook_name]

                for head in range(self.model.cfg.n_heads):
                    slc = (slice(start, end), slice(None), layer, head)

                    # compute steering vector
                    steering = z_corrupt[..., head, :] - z_clean[..., head, :]

                    # store clean score
                    if "logit_clean_score" in self.raw_tensors:
                        self.raw_tensors["logit_clean_score"][slc] = clean_score.cpu()

                    # get intervention metrics
                    head_metrics = self._compute_metrics_for_head(
                        head_indices=(layer, head),
                        clean_tokens=clean_tokens,
                        clean_cache_for_freezing=norm_cache,     # for downstream freezing
                        steering=steering
                    )
                    for name, val in head_metrics.items():
                        if name in self.raw_tensors:
                            self.raw_tensors[name][slc] = val.cpu()

                    # store DE if available
                    if per_head_de is not None:
                        self.raw_tensors["direct_effects"][slc] = per_head_de[layer, head].cpu()

            pbar.update(1)
        pbar.close()

    def compute_aggregated_metrics(self):
        """Computes condensed (top DE percentile) and full (average) metrics."""
        if not self.raw_tensors or self.config.prompt_len <= 1:
            print("Raw tensors not computed or prompt_len <= 1. Skipping aggregation.")
            return

        print("Computing aggregated metrics...")

        # --- Process condensed tensors (top DE instances) --- #
        # Check if DE was computed and is needed
        de_available = (
            "direct_effects" in self.raw_tensors
            and self.raw_tensors["direct_effects"].numel() > 0
            and not torch.isnan(self.raw_tensors["direct_effects"]).all()
        )

        can_compute_condensed = self.config.num_top_instances > 0 and de_available

        if can_compute_condensed:
            print(
                f"Selecting top {self.config.percentile*100:.2f}% DE instances "
                f"({self.config.num_top_instances})..."
            )
            direct_effects_all = (
                self.raw_tensors["direct_effects"].nan_to_num().to(self.config.device)
            )

            for layer in tqdm(range(self.model.cfg.n_layers), desc="Agg. Condensed"):
                for head in range(self.model.cfg.n_heads):
                    try:
                        # Get top (batch_idx, pos_idx) for this head based on DE
                        top_indices = topk_of_Nd_tensor(
                            direct_effects_all[
                                ..., layer, head
                            ],  # Shape (batch, seq-1)
                            k=min(
                                self.config.num_top_instances,
                                direct_effects_all[..., layer, head].numel(),
                            ),  # Ensure k is not too large
                        )
                        if not top_indices:
                            continue  # Skip if no indices found

                    except Exception as e:
                        print(f"Error getting top_indices L{layer}H{head}: {e}. Skip.")
                        continue

                    # Average metrics over these top indices
                    for name, raw_tensor in self.raw_tensors.items():
                        condensed_name = f"condensed_{name}"
                        if condensed_name in self.condensed_tensors:
                            try:
                                # Gather data for top instances (move to device for calc)
                                data_for_top = (
                                    torch.stack(
                                        [
                                            raw_tensor[b, p, layer, head]
                                            for b, p in top_indices  # Indices are (batch, pos)
                                        ]
                                    )
                                    .to(self.config.device)
                                    .float()
                                )  # Ensure float

                                # Use nanmean for robustness
                                agg_val = (
                                    torch.nanmean(data_for_top)
                                    if torch.isnan(data_for_top).any()
                                    else torch.mean(data_for_top)
                                )
                                self.condensed_tensors[condensed_name][
                                    layer, head
                                ] = agg_val
                            except IndexError:
                                print(
                                    f"Index error aggregating cond {name} L{layer}H{head}. Likely due to invalid top_indices. Setting NaN."
                                )
                                self.condensed_tensors[condensed_name][
                                    layer, head
                                ] = torch.nan
                            except Exception as e:
                                print(
                                    f"Error aggregating cond {name} L{layer}H{head}: {e}"
                                )
                                self.condensed_tensors[condensed_name][
                                    layer, head
                                ] = torch.nan

            del direct_effects_all
            if self.config.device.type == "cuda":
                torch.cuda.empty_cache()

        elif self.config.num_top_instances == 0:
            print("Skipping condensed calculation: num_top_instances is 0.")
        else:
            print(
                "Skipping condensed calculation: 'direct_effects' missing, empty, or all NaN."
            )

        # --- Calculate full tensors (averages over all instances) --- #
        print("Aggregating Full...")
        for name, raw_tensor in tqdm(self.raw_tensors.items(), desc="Aggregating Full"):
            full_name = f"full_{name}"
            if full_name in self.full_tensors:
                try:
                    # Move tensor to device for aggregation, ensure float
                    raw_tensor_device = raw_tensor.to(self.config.device).float()
                    # Mean over batch and position (dims 0, 1)
                    # Check for all NaNs before calculating mean to avoid warning
                    if torch.isnan(raw_tensor_device).all():
                        agg_val = torch.nan
                    else:
                        agg_val = torch.nanmean(raw_tensor_device, dim=(0, 1))

                    self.full_tensors[full_name] = agg_val
                except Exception as e:
                    print(f"Error aggregating full {name}: {e}")
                    # Fill with NaN on error
                    self.full_tensors[full_name] = torch.full_like(
                        self.full_tensors[full_name], torch.nan
                    )

        if self.config.device.type == "cuda":
            torch.cuda.empty_cache()
        print("Finished computing aggregated metrics.")

    def get_results(self) -> Dict[str, Tensor]:
        """Returns all computed aggregated tensors."""
        results = {**self.full_tensors}
        # Add condensed tensors if they were computed
        if self.config.num_top_instances > 0:
            # Check if at least one condensed tensor has non-NaN values before adding
            condensed_computed = any(
                name in self.condensed_tensors and not torch.isnan(tensor).all()
                for name, tensor in self.condensed_tensors.items()
            )
            if condensed_computed:
                results.update(self.condensed_tensors)

        # Return only tensors that have data and are not entirely NaN
        return {
            k: v
            for k, v in results.items()
            if v.numel() > 0 and not torch.isnan(v).all()
        }


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
        """Loads the HookedTransformer model."""
        # Add dtype configuration if needed, e.g., torch.float16 for memory saving
        model = HookedTransformer.from_pretrained(
            self.config.model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=False,  # Keep False unless specific reason
            device=self.config.device,
            # dtype=torch.float16 # Uncomment if using float16
        )
        # Ensure necessary hook points are available
        model.set_use_attn_result(True)  # Needed for hook_z
        model.set_use_split_qkv_input(True)  # Needed for ln1/ln2 hooks used in freezing
        model.set_use_hook_mlp_in(True)  # Needed for ln2 hook
        # Ensure normalization hooks are generated if using them
        # These flags might need adjustment based on exact hook points used in freeze_hook_fn
        model.cfg.use_hook_ln_pre = True
        model.cfg.use_hook_normalized = (
            True  # Crucial if freezing `*.hook_normalized` points
        )

        return model

    def run(self) -> Dict[str, Tensor]:
        """Runs the full self-repair computation pipeline."""
        try:
            self.calculator.compute_raw_metrics(self.data_loader)
            self.calculator.compute_aggregated_metrics()
            self.results = self.calculator.get_results()

            if self.config.save_results:
                self.save_results()
        except Exception as e:
            print("\n--- Pipeline execution failed ---")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            print("---------------------------------")

        print("Pipeline finished.")
        return self.results

    def save_results(self):
        """Saves the computed aggregated tensors to a single pickle file."""
        if not self.results:
            print("No results to save.")
            return

        print(f"Saving results to {self.config.save_folder}...")
        save_dir = self.config.save_folder
        os.makedirs(save_dir, exist_ok=True)

        # Create a dictionary with all tensors moved to CPU
        results_dict = {name: tensor.cpu() for name, tensor in self.results.items()}
        
        # Construct filename including percentile info
        file_path = os.path.join(
            save_dir,
            f"{self.config.percentile_str}{self.config.safe_model_name}_all_results.pkl",
        )
        
        try:
            with open(file_path, "wb") as f:
                pickle.dump(results_dict, f)
            print(f"Finished saving all results to {file_path}")
        except Exception as e:
            print(f"Error saving results to {file_path}: {e}")

    def load_results(self, file_path: str) -> Dict[str, Tensor]:
        """Loads the computed aggregated tensors from a single pickle file."""
        with open(file_path, "rb") as f:
            results_dict =  pickle.load(f)
        self.results = {name: tensor.to(self.config.device) for name, tensor in results_dict.items()}
        return self.results


# --- Example Usage ---

if __name__ == "__main__":
    # 1. Configure the pipeline
    config = SelfRepairConfigNew(
        model_name="google/gemma-3-1b-it",
        dataset_name="pile",
        batch_size=2,               # Adjust based on memory
        total_tokens_in_data=1000,  # Smaller run for quicker example
        prompt_len=50,
        percentile=0.05,
        # total_tokens_in_data=1000,
        save_results=True,
        save_folder="data/output/new_self_repair_results_runhooks/",  # Changed folder
        metrics=[
            "direct_effects",
            "logit_clean_score",
            "logit_ablated_score",
            "logit_iso_ablated_score",
            "self_repair_new",
        ],
        # device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Explicitly set device
    )

    # 2. Create and run the pipeline
    pipeline = SelfRepairPipelineNew(config)
    results = pipeline.run()

    # 3. Access results using pandas for better display
    import pandas as pd
    
    print("\n--- Results Summary ---")
    if not results:
        print("No results were generated (pipeline might have failed).")
    else:
        print("Computed results keys:", list(results.keys()))
        
        # Create a summary dataframe
        summary_data = []
        for key in results.keys():
            tensor = results[key]
            shape = tensor.shape if hasattr(tensor, 'shape') else None
            has_values = tensor.numel() > 0 if hasattr(tensor, 'numel') else False
            example_00 = tensor[0, 0].item() if has_values else None
            example_22 = tensor[2, 2].item() if has_values else None
            example_44 = tensor[4, 4].item() if has_values else None
            
            summary_data.append({
                'Metric': key,
                'Shape': str(shape),
                'Has Values': has_values,
                'Example (L0H0)': f"{example_00:.4f}" if example_00 is not None else "N/A",
                'Example (L2H2)': f"{example_22:.4f}" if example_22 is not None else "N/A",
                'Example (L4H4)': f"{example_44:.4f}" if example_44 is not None else "N/A"
            })
        
        # Display as a clean table
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    print("----------------------")
