import transformer_lens as ts


# Function to generate a dataset function
def dataset_generator(dataset_tokens, batch_size, prompt_len):
    total_prompts = dataset_tokens.shape[0]
    num_batches = total_prompts // batch_size

    # print(num_batches)
    for batch_idx in range(num_batches):
        clean_batch_offset = batch_idx * batch_size
        start_clean_prompt = clean_batch_offset
        end_clean_prompt = clean_batch_offset + batch_size

        corrupted_batch_offset = (batch_idx + 1) * batch_size
        start_corrupted_prompt = corrupted_batch_offset
        end_corrupted_prompt = corrupted_batch_offset + batch_size

        clean_tokens = dataset_tokens[start_clean_prompt:end_clean_prompt, :prompt_len]
        corrupted_tokens = dataset_tokens[
            start_corrupted_prompt:end_corrupted_prompt, :prompt_len
        ]
        # print(corrupted_tokens.shape[0])
        if corrupted_tokens.shape[0] == 0:
            corrupted_tokens = dataset_tokens[
                :batch_size, :prompt_len
            ]  # loop it back around

        yield batch_idx, clean_tokens, corrupted_tokens


def prepare_dataset(
    model,
    device,
    total_tokens_in_data: int,
    batch_size,
    prompt_len,
    padding: bool,
    dataset_name="pile",  # NOTE: pile is huge
):
    """
    returns the dataset, and the number of prompts in this dataset
    """
    dataset = ts.utils.get_dataset(dataset_name)

    if not padding:
        new_dataset = ts.utils.tokenize_and_concatenate(
            dataset, model.tokenizer, max_length=prompt_len
        )
        all_dataset_tokens = new_dataset["tokens"].to(device)
    else:
        print("Not complete yet")
        all_dataset_tokens = model.to_tokens(dataset["text"]).to(device)
        PAD_TOKEN = model.to_tokens(model.tokenizer.pad_token)[-1, -1].item()

    assert len(all_dataset_tokens.shape) == 2
    total_prompts = total_tokens_in_data // (prompt_len)
    num_batches = total_prompts // batch_size

    if num_batches <= 1:
        raise ValueError(
            "Need to have more than 2 batches for corrupt prompt gen to work"
        )

    # Create the generator
    dataset = dataset_generator(
        all_dataset_tokens[:total_prompts], batch_size, prompt_len
    )

    return dataset, num_batches
