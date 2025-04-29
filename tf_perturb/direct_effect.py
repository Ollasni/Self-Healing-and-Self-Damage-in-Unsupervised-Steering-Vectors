from typing import Union, List, Optional, Callable, Tuple, Dict, Literal, Set
from tqdm import tqdm
from pathlib import Path

import numpy as np
import itertools
from matplotlib import pyplot as plt

import einops
import torch
from torch import Tensor


from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from fancy_einsum import einsum
import transformer_lens as ts
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
    patching,
)


torch.set_grad_enabled(False)


def residual_stack_to_direct_effect(
    residual_stack: Float[Tensor, "... batch pos d_model"],
    effect_directions: Float[Tensor, "batch pos d_model"],
    apply_last_ln=True,
    scaling_cache=None,
) -> Float[Tensor, "... batch pos_mins_one"]:
    """
    Gets the direct effect towards a direction for a given stack of components in the residual stream.
    NOTE: IGNORES THE VERY LAST PREDICTION AND FIRST CLEAN TOKEN; WE DON'T KNOW THE ACTUAL PREDICTED ANSWER FOR IT!

    residual_stack: [... batch pos d_model] components of d_model vectors to measure direct effect from
    effect_directions: [batch pos d_model] vectors in d_model space that correspond to 'direct effect'
    scaling_cache = the cache to use for the scaling; defaults to the global clean cache
    """

    if scaling_cache is None:
        raise ValueError("scaling_cache cannot be None")

    scaled_residual_stack = (
        scaling_cache.apply_ln_to_stack(residual_stack, layer=-1, has_batch_dim=True)
        if apply_last_ln
        else residual_stack
    )

    # remove the last prediction, and the direction of the zeroth token
    scaled_residual_stack = scaled_residual_stack[..., :, :-1, :]
    effect_directions = effect_directions[:, 1:, :]

    return einops.einsum(
        scaled_residual_stack,
        effect_directions,
        "... batch pos d_model, batch pos d_model -> ... batch pos",
    )


def collect_direct_effect(
    de_cache: ActivationCache = None,
    correct_tokens: Float[Tensor, "batch seq_len"] = None,
    model: HookedTransformer = None,
    collect_individual_neurons=False,
    cache_for_scaling: ActivationCache = None,
) -> Tuple[
    Float[Tensor, "n_layer n_head batch pos_minus_one"],
    Float[Tensor, "n_layer batch pos_minus_one"],
]:  # , Float[Tensor, "n_layer d_mlp batch pos_minus_one"]]:
    """
    Given a cache of activations, and a set of correct tokens, returns the direct effect of each head and neuron on each token.

    returns tuple of tensors of per-head, per-mlp-layer, per-neuron of direct effects

    cache: cache of activations from the model
    correct_tokens: [batch, seq_len] tensor of correct tokens
    title: title of the plot (relavant if display == True)
    display: whether to display the plot or return the data; if False, returns [head, pos] tensor of direct effects
    cache_for_scaling: the cache to use for the scaling; defaults to the global clean cache
    """

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if de_cache is None or correct_tokens is None or model is None:
        raise ValueError("de_cache, correct_tokens, and model must not be None")

    if cache_for_scaling is None:
        cache_for_scaling = de_cache

    token_residual_directions: Float[Tensor, "batch seq_len d_model"] = (
        model.tokens_to_residual_directions(correct_tokens)
    )

    # get the direct effect of heads by positions
    clean_per_head_residual: Float[Tensor, "head batch seq d_model"] = (
        de_cache.stack_head_results(layer=-1, return_labels=False, apply_ln=False)
    )

    # print(clean_per_head_residual.shape)
    per_head_direct_effect: Float[Tensor, "heads batch pos_minus_one"] = (
        residual_stack_to_direct_effect(
            clean_per_head_residual,
            token_residual_directions,
            True,
            scaling_cache=cache_for_scaling,
        )
    )

    per_head_direct_effect = einops.rearrange(
        per_head_direct_effect,
        "(n_layer n_head) batch pos -> n_layer n_head batch pos",
        n_layer=model.cfg.n_layers,
        n_head=model.cfg.n_heads,
    )
    # assert per_head_direct_effect.shape == (model.cfg.n_heads * model.cfg.n_layers, tokens.shape[0], tokens.shape[1])

    # get the outputs of the neurons
    direct_effect_mlp: Float[Tensor, "n_layer d_mlp batch pos_minus_one"] = torch.zeros(
        (
            model.cfg.n_layers,
            model.cfg.d_mlp,
            correct_tokens.shape[0],
            correct_tokens.shape[1] - 1,
        )
    )

    # iterate over every neuron to avoid memory issues
    if collect_individual_neurons:
        for neuron in range(model.cfg.d_mlp):
            single_neuron_output: Float[Tensor, "n_layer batch pos d_model"] = (
                de_cache.stack_neuron_results(
                    layer=-1,
                    neuron_slice=(neuron, neuron + 1),
                    return_labels=False,
                    apply_ln=False,
                )
            )
            direct_effect_mlp[:, neuron, :, :] = residual_stack_to_direct_effect(
                single_neuron_output,
                token_residual_directions,
                scaling_cache=cache_for_scaling,
            )
    # get per mlp layer effect
    all_layer_output: Float[Tensor, "n_layer batch pos d_model"] = torch.zeros(
        (
            model.cfg.n_layers,
            correct_tokens.shape[0],
            correct_tokens.shape[1],
            model.cfg.d_model,
        )
    ).to(device)
    for layer in range(model.cfg.n_layers):
        all_layer_output[layer, ...] = de_cache[f"blocks.{layer}.hook_mlp_out"]

    all_layer_direct_effect: Float["n_layer batch pos_minus_one"] = (
        residual_stack_to_direct_effect(
            all_layer_output, token_residual_directions, scaling_cache=cache_for_scaling
        )
    )

    per_head_direct_effect = per_head_direct_effect.to(device)
    all_layer_direct_effect = all_layer_direct_effect.to(device)
    direct_effect_mlp = direct_effect_mlp.to(device)

    if collect_individual_neurons:
        return per_head_direct_effect, all_layer_direct_effect, direct_effect_mlp
    else:
        return per_head_direct_effect, all_layer_direct_effect


def get_correct_logit_score(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_tokens: Float[Tensor, "batch seq"],
):
    """
    Returns the logit of the next token

    If per_prompt=True, return the array of differences rather than the average.
    """
    smaller_logits = logits[:, :-1, :]
    smaller_correct = clean_tokens[:, 1:].unsqueeze(-1)
    answer_logits: Float[Tensor, "batch 2"] = smaller_logits.gather(
        dim=-1, index=smaller_correct
    )
    return answer_logits.squeeze()  # get rid of last index of size one


# test with this
# a = torch.tensor([[[0,1,2,3,4], [10,11,12,13,14], [100,101,120,103,140]],
#                   [[10,999,2,3,4], [110,191,120,13,14], [1100,105,120,103,140]]])
# get_correct_logit_score(a, clean_tokens = torch.tensor([[3, 2, 4], [0,1,2]]))


def shuffle_owt_tokens_by_batch(
    owt_tokens: torch.Tensor, offset_shuffle=2
) -> torch.Tensor:
    """Shuffles the prompts in a batch by just moving them by an offset."""
    # Roll the batch dimension by the specified offset
    if offset_shuffle == 0:
        print("Warning: offset_shuffle = 0, so no shuffling is happening")

    shuffled_owt_tokens = torch.roll(owt_tokens, shifts=offset_shuffle, dims=0)
    return shuffled_owt_tokens


def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    """
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    """
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(ts.utils.to_numpy(i), tensor.shape)).T.tolist()


def get_projection(
    from_vector: Float[Tensor, "batch d_model"],
    to_vector: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "batch d_model"]:
    assert from_vector.shape == to_vector.shape
    assert from_vector.ndim == 2

    dot_product = einops.einsum(
        from_vector, to_vector, "batch d_model, batch d_model -> batch"
    )
    # length_of_from_vector = einops.einsum(from_vector, from_vector, "batch d_model, batch d_model -> batch")
    length_of_vector = einops.einsum(
        to_vector, to_vector, "batch d_model, batch d_model -> batch"
    )

    projected_lengths = (dot_product) / (length_of_vector)
    projections = to_vector * einops.repeat(
        projected_lengths, "batch -> batch d_model", d_model=to_vector.shape[-1]
    )
    return projections


def get_3d_projection(
    from_vector: Float[Tensor, "batch seq d_model"],
    to_vector: Float[Tensor, "batch seq d_model"],
) -> Float[Tensor, "batch seq d_model"]:
    assert from_vector.shape == to_vector.shape
    assert from_vector.ndim == 3

    dot_product = einops.einsum(
        from_vector, to_vector, "batch seq d_model, batch seq d_model -> batch seq"
    )
    length_of_vector = einops.einsum(
        to_vector, to_vector, "batch seq d_model, batch seq d_model -> batch seq"
    )

    projected_lengths = (dot_product) / (length_of_vector)
    projections = to_vector * einops.repeat(
        projected_lengths, "batch seq -> batch seq d_model", d_model=to_vector.shape[-1]
    )
    return projections


#  Code to first intervene by subtracting output in residual stream
def add_vector_to_resid(
    original_resid_stream: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    vector: Float[Tensor, "batch d_model"],
    positions=Union[Float[Tensor, "batch"], int],
) -> Float[Tensor, "batch n_head pos pos"]:
    """
    Hook that adds vector into residual stream at position
    """
    assert len(original_resid_stream.shape) == 3
    assert len(vector.shape) == 2
    assert original_resid_stream.shape[0] == vector.shape[0]
    assert original_resid_stream.shape[2] == vector.shape[1]

    if isinstance(positions, int):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        positions = torch.tensor([positions] * original_resid_stream.shape[0]).to(
            device
        )

    expanded_positions = einops.repeat(
        positions, "batch -> batch 1 d_model", d_model=vector.shape[1]
    )
    resid_stream_at_pos = torch.gather(original_resid_stream, 1, expanded_positions)
    resid_stream_at_pos = einops.rearrange(
        resid_stream_at_pos, "batch 1 d_model -> batch d_model"
    )

    resid_stream_at_pos = resid_stream_at_pos + vector
    for i in range(original_resid_stream.shape[0]):
        original_resid_stream[i, positions[i], :] = resid_stream_at_pos[i]
    return original_resid_stream


#
def add_vector_to_all_resid(
    original_resid_stream: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    vector: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "batch n_head pos pos"]:
    """
    Hook that just adds a vector to the entire residual stream
    """
    assert len(original_resid_stream.shape) == 3
    assert original_resid_stream.shape == vector.shape

    original_resid_stream = original_resid_stream + vector
    return original_resid_stream


def replace_output_hook(
    original_output: Float[Tensor, "batch seq head d_model"],
    hook: HookPoint,
    new_output: Float[Tensor, "batch seq d_model"],
    head: int,
) -> Float[Tensor, "batch seq d_model"]:
    """
    Hook that replaces the output of a head with a new output
    """

    assert len(original_output.shape) == 4
    assert len(new_output.shape) == 3
    assert original_output.shape[0] == new_output.shape[0]
    assert original_output.shape[1] == new_output.shape[1]
    assert original_output.shape[3] == new_output.shape[2]

    original_output[:, :, head, :] = new_output

    return original_output


def replace_output_of_specific_batch_pos_hook(
    original_output: Float[Tensor, "batch seq head d_model"],
    hook: HookPoint,
    new_output: Float[Tensor, "d_model"],
    head: int,
    batch: int,
    pos: int,
) -> Float[Tensor, "batch seq d_model"]:
    """
    Hook that replaces the output of a head on a batch/pos with a new output
    """
    # print(original_output.shape)
    # print(new_output.shape)
    assert len(original_output.shape) == 4
    assert len(new_output.shape) == 1
    assert original_output.shape[-1] == new_output.shape[-1]

    original_output[batch, pos, head, :] = new_output

    return original_output


def replace_output_of_specific_MLP_batch_pos_hook(
    original_output: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    new_output: Float[Tensor, "d_model"],
    batch: int,
    pos: int,
) -> Float[Tensor, "batch seq d_model"]:
    """
    Hook that replaces the output of a MLP layer on a batch/pos with a new output
    """

    assert len(original_output.shape) == 3
    assert len(new_output.shape) == 1
    assert original_output.shape[-1] == new_output.shape[-1]

    original_output[batch, pos, :] = new_output
    return original_output


def get_item_hook(item, hook: HookPoint, storage: list):
    """
    Hook that just returns this specific item.
    """
    storage.append(item)
    return item


def replace_model_component_completely(
    model_comp,
    hook: HookPoint,
    new_model_comp,
):
    if isinstance(model_comp, torch.Tensor):
        model_comp[:] = new_model_comp
    return new_model_comp


def get_single_correct_logit(
    logits: Float[Tensor, "batch pos d_vocab"],
    batch: int,
    pos: int,
    tokens: Float[Tensor, "batch pos"],
):
    """
    get the correct logit at a specific batch and position (of the next token)
    """

    correct_next_token = tokens[batch, pos + 1]
    return logits[batch, pos, correct_next_token]
