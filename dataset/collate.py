import torch

# TODO rewrite this collate function 
def collate_fn(examples):
    if examples[0].get("cond_pixel_values") is not None:
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        cond_pixel_values = None
    if examples[0].get("subject_pixel_values") is not None: 
        subject_pixel_values = torch.stack([example["subject_pixel_values"] for example in examples])
        subject_pixel_values = subject_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        subject_pixel_values = None

    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    # token_ids_clip = torch.stack([torch.tensor(example["token_ids_clip"]) for example in examples])
    # token_ids_t5 = torch.stack([torch.tensor(example["token_ids_t5"]) for example in examples])

    return {
        "cond_pixel_values": cond_pixel_values,
        "subject_pixel_values": subject_pixel_values,
        "pixel_values": target_pixel_values,
    }