# %%
import autoroot 
import autorootcwd
import logging
import random
import traceback
from io import BytesIO
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import transformers
from einops import rearrange
from loguru import logger
from PIL import Image
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset.mask_gen import get_mask_generator
from dataset.aspect_ratios import get_aspect_ratio_map, get_closest_ratio
from dataset.streaming_utils import make_streams

DEFAULT_MASK_GEN_KWARGS = dict(
    irregular_kwargs=dict(max_angle=4 * 5, max_len=200 * 5, max_width=100 * 5, max_times=1, min_times=1),
    box_kwargs=dict(margin=10, bbox_min_size=30 * 5, bbox_max_size=150 * 5, max_times=1, min_times=1),
)


class StreamingImageCaptionInpaintDataset(StreamingDataset):
    """A streaming dataset for image inpainting tasks with captions.

    This dataset loads images and captions, generates masks for inpainting,
    and prepares data for image inpainting models. It supports aspect ratio
    bucketing, various mask generation strategies (irregular, box, dumb), and
    caption dropping.
    """

    def __init__(
        self,
        # tokenizer: None | List[transformers.PreTrainedTokenizer] | transformers.PreTrainedTokenizer = None,
        streams: None | Sequence[Stream] = None,
        remote: None | str = None,
        local: None | str = None,
        caption_drop_prob: float = 0.0,
        microcond_drop_prob: float = 0.0,
        item_key: str = "__key__",
        image_key: str = "image",
        caption_key: str = "caption",
        caption_metadata: None | pd.DataFrame = None,
        sdxl_conditioning: bool = False,
        aspect_ratio: str = "ASPECT_RATIO_1024",
        center_crop: bool = False,
        irregular_mask_prob: float = 1.0,
        box_mask_prob: float = 1.0,
        switch_to_dumb_mask_prob: float = 0.25,
        cond_size: int = 512,
        output_keys: Dict[str, str] = {"image": "pixel_values", "mask": "mask_values", "prompt": "prompt", "img_hw": "img_hw", "aspect_ratio": "aspect_ratio"},
        **streaming_kwargs,
    ):
        """Initializes the StreamingImageCaptionInpaintDataset.

        Args:
            streams (None | Sequence[Stream], optional): Sequence of Stream objects.
                Defaults to None.
            remote (None | str, optional): Remote path for streaming. Defaults to None.
            local (None | str, optional): Local path for caching. Defaults to None.
            caption_drop_prob (float, optional): Probability of dropping captions.
                Defaults to 0.0.
            microcond_drop_prob (float, optional): Probability of dropping micro-conditioning.
                (Not currently used). Defaults to 0.0.
            item_key (str, optional): Key for item identifier in samples. Defaults to "__key__".
            image_key (str, optional): Key for image data in samples. Defaults to "image".
            caption_key (str, optional): Key for caption in `caption_metadata`.
                Defaults to "caption".
            caption_metadata (None | pd.DataFrame, optional): DataFrame with captions.
                Defaults to None.
            sdxl_conditioning (bool, optional): Enable SDXL conditioning.
                (Not supported). Defaults to False.
            aspect_ratio (str, optional): Aspect ratio map name (e.g., "ASPECT_RATIO_1024").
                Defaults to "ASPECT_RATIO_1024".
            center_crop (bool, optional): Use center crop if True, else random crop.
                Defaults to False.
            irregular_mask_prob (float, optional): Probability for irregular mask generator.
                Defaults to 1.0.
            box_mask_prob (float, optional): Probability for box mask generator.
                Defaults to 1.0.
            switch_to_dumb_mask_prob (float, optional): Probability to switch to a simple
                ("dumb") mask generator. Defaults to 0.25.
            cond_size (int, optional): Size of the conditioning image. Defaults to 64.
            output_keys (Dict[str, str], optional): Mapping for output dictionary keys.
                Defaults to a predefined mapping.
            **streaming_kwargs: Additional arguments for `StreamingDataset`.
        """
        assert sdxl_conditioning is False, "SDXL conditioning is not supported yet"

        # Set defaults for vision-friendly streaming args.
        streaming_kwargs.setdefault("shuffle_block_size", 1 << 18)
        streaming_kwargs.setdefault("shuffle_algo", "py1s")
        super().__init__(streams=streams, remote=remote, local=local, **streaming_kwargs)

        self.center_crop = center_crop
        # self.tokenizer = tokenizer
        self.caption_drop_prob = caption_drop_prob
        self.microcond_drop_prob = microcond_drop_prob
        self.item_key = item_key
        self.image_key = image_key
        self.caption_key = caption_key
        self.caption_metadata = caption_metadata
        self.sdxl_conditioning = sdxl_conditioning
        self.aspect_ratio = get_aspect_ratio_map(aspect_ratio)
        mask_gen_kwargs = DEFAULT_MASK_GEN_KWARGS
        mask_gen_kwargs["irregular_proba"] = irregular_mask_prob
        mask_gen_kwargs["box_proba"] = box_mask_prob
        self.mixed_mask_gen = get_mask_generator(kind="mixed", kwargs=mask_gen_kwargs)
        self.dumb_mask_gen = get_mask_generator(kind="dumb", kwargs={})
        self.switch_to_dumb_mask_prob = switch_to_dumb_mask_prob
        self.output_keys = output_keys
        self.norm_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        # Define conditioning transforms in __init__ since they're constant
        self.cond_train_transforms = T.Compose([
            T.Resize((cond_size, cond_size), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop((cond_size, cond_size)), 
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    @staticmethod
    def get_closest_ratio(height: float, width: float, ratios: dict):
        """Finds the closest aspect ratio from a predefined set.

        Args:
            height (float): The height of the image.
            width (float): The width of the image.
            ratios (dict): A dictionary where keys are string representations of aspect
                ratios (e.g., "1.0") and values are the target (height, width) tuples.

        Returns:
            tuple[tuple[int, int], float]:
                - The target (height, width) tuple for the closest aspect ratio.
                - The float value of the closest aspect ratio.
        """
        aspect_ratio = height / width
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
        return ratios[closest_ratio], float(closest_ratio)

    def get(self, index):
        """Retrieves and processes a single sample for inpainting.

        Fetches an image and caption, resizes/crops the image, generates a mask,
        and normalizes the image and mask.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the processed sample, with keys defined
                by `self.output_keys`. Common keys include:
                - "image" or "pixel_values": The transformed image tensor (C, H, W).
                - "prompt": The caption string.
                - "mask" or "mask_values": The generated mask tensor (C, H, W).
                - "img_hw": Original image (height, width) as a tensor.
                - "aspect_ratio": The closest float aspect ratio.
        """
        sample = super().__getitem__(index)
        data_info = {}

        item_key = sample[self.item_key]
        try:
            prompt = self.caption_metadata.query(f"{self.item_key} == @item_key")[self.caption_key].item()
        except Exception as e:
            logger.warning(f"Error getting caption for item {item_key}: {e}. Returning empty string as caption.")
            prompt = ""

        if random.random() < self.caption_drop_prob:
            prompt = ""

        img: Image.Image = sample[self.image_key]
        original_width, original_height = img.size

        closest_size, closest_ratio = self.get_closest_ratio(original_height, original_width, self.aspect_ratio)
        closest_size = list(map(lambda x: int(x), closest_size))

        data_info[self.output_keys.get("img_hw", "img_hw")] = torch.tensor([original_height, original_width], dtype=torch.float32)
        data_info[self.output_keys.get("aspect_ratio", "aspect_ratio")] = closest_ratio

        if closest_size[0] / original_height > closest_size[1] / original_width:
            resize_size = closest_size[0], int(original_width * closest_size[0] / original_height)
        else:
            resize_size = int(original_height * closest_size[1] / original_width), closest_size[1]

        if self.center_crop:
            self.resize_transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB")),
                    T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(closest_size),
                ]
            )
        else:
            self.resize_transform = T.Compose(
                [
                    T.Lambda(lambda img: img.convert("RGB")),
                    T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomCrop(closest_size),
                ]
            )
        pil_img = self.resize_transform(img)
        np_img_transposed = np.asanyarray(pil_img).transpose(2, 0, 1).astype(np.uint8)
        
        if random.random() < self.switch_to_dumb_mask_prob:
            np_mask = self.dumb_mask_gen(np_img_transposed)
        else:
            np_mask = self.mixed_mask_gen(np_img_transposed)

        np_mask = np.asanyarray(np_mask).astype(np.float32)
        np_mask = rearrange(np_mask, "c h w -> h w c")
        np_mask[np_mask < 0.5] = 0.0
        np_mask[np_mask >= 0.5] = 1.0
        # Create masked image by applying mask to PIL image
        pil_np = np.array(pil_img)
        # Broadcast mask to match image channels
        assert pil_np.shape[:2] == np_mask.shape[:2], f"Shape mismatch in height/width: img {pil_np.shape[:2]}, mask {np_mask.shape[:2]}"
        masked_img = pil_np * (1 - np_mask)
        masked_img = Image.fromarray(masked_img.astype(np.uint8))

        tensor_mask = torch.from_numpy(np_mask).contiguous()
        tensor_mask = rearrange(tensor_mask, "h w c -> c h w")

        tensor_img = self.norm_transform(pil_img)
        tensor_masked_img = self.norm_transform(masked_img)

        data_info[self.output_keys.get("image", "image")] = tensor_img.contiguous()
        data_info[self.output_keys.get("prompt", "prompt")] = prompt
        data_info[self.output_keys.get("mask", "mask")] = tensor_mask.contiguous()
        # Apply conditioning transforms to masked image
        cond_pixel_values = self.cond_train_transforms(masked_img)
        data_info[self.output_keys.get("cond_pixel_values", "cond_pixel_values")] = cond_pixel_values.contiguous()

        return data_info
        return data_info
        return data_info
        return data_info

    def __getitem__(self, index):
        """Alias for the `get` method."""
        return self.get(index)


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # batch is a list of dicts, each with keys:
    # ["img_hw", "aspect_ratio", "pixel_values", "mask_values", "masked_image", "prompt"]
    return {
        "img_hw":          torch.stack([sample["img_hw"]          for sample in batch], dim=0),  # (B, 2)
        "aspect_ratio":    torch.tensor([sample["aspect_ratio"]    for sample in batch], dtype=torch.float32),  # (B,)
        "pixel_values":    torch.stack([sample["pixel_values"]    for sample in batch], dim=0),  # (B, 3, H, W)
        "mask_values":     torch.stack([sample["mask_values"]     for sample in batch], dim=0),  # (B, 1, H, W)
        "masked_image":    torch.stack([sample["cond_pixel_values"]    for sample in batch], dim=0),  # (B, 3, H, W)
        "prompt":       [sample["prompt"]                   for sample in batch],         # List[str] of length B
    }


# %%
if __name__ == "__main__":
    from streaming import Stream, StreamingDataLoader
    from streaming.base.util import clean_stale_shared_memory
    import tempfile

    clean_stale_shared_memory()
    # Create streams
    STREAMS = make_streams(
        remote=[
            "/mnt/dashtoon_data/data_hub/gallery-dl/pinterest_mds_shards/aspect-ratio-0.6",
            # "/mnt/dashtoon_data/data_hub/gallery-dl/pinterest_mds_shards/aspect-ratio-0.52",
            # "/mnt/dashtoon_data/data_hub/gallery-dl/pinterest_mds_shards/aspect-ratio-0.57",
            # "/mnt/dashtoon_data/data_hub/gallery-dl/pinterest_mds_shards/aspect-ratio-0.68",
        ],
        local=tempfile.mkdtemp(prefix="streaming_cache_"),
        choose=[50],
    )
    print(STREAMS)

    # Load caption metadata
    meta_paths = [
        # "/mnt/dashtoon_data/data_hub/playbook-dataset/meta_caption_info_rewritten.jsonl",
        "/mnt/dashtoon_data/data_hub/gallery-dl/pinterest_meta_caption_info_rewritten.jsonl",
    ]

    caption_meta = pd.concat(pd.read_json(p, orient="records", lines=True) for p in meta_paths)
    caption_meta = caption_meta.reset_index(drop=True, inplace=False)
    # caption_meta = caption_meta.iloc[:50].reset_index(drop=True)

    # Create dataset and dataloader
    streaming_kwargs = dict(shuffle=True, batch_size=4, batching_method="per_stream")
    dataset = StreamingImageCaptionInpaintDataset(
        streams=STREAMS,
        item_key="__key__",
        image_key="image",
        caption_key="caption",
        caption_metadata=caption_meta,
        sdxl_conditioning=False,
        aspect_ratio="ASPECT_RATIO_1024",
        center_crop=False,
        irregular_mask_prob=1.0,
        box_mask_prob=1.0,
        switch_to_dumb_mask_prob=0.25,
        **streaming_kwargs,
    )
    dataloader = StreamingDataLoader(
        dataset, 
        batch_size=4, 
        num_workers=32, 
        prefetch_factor=2, 
        pin_memory=True, 
        # collate_fn=custom_collate_fn
        )

    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    num_batches = len(dataloader)
    print(f"Number of batches: {num_batches}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of workers: {dataloader.num_workers}")
    print("\nSample batch contents:")
    
    # import tqdm
    # for batch in tqdm(dataloader, desc="Caching latents"):
    #     print(batch)
    #     break 

    # Iterate through batches and save sample images
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}/{num_batches}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: tensor shape {v.shape}")
                if k == "pixel_values":
                    # Convert pixel values tensor to images and save
                    for i in range(v.shape[0]):
                        img = v[i].permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save(f"pixel_values_batch{batch_idx}_sample{i}.png")
                elif k == "mask_values":
                    # Convert mask values tensor to images and save 
                    for i in range(v.shape[0]):
                        mask = v[i].squeeze().cpu().numpy()
                        mask = (mask * 255).astype(np.uint8)
                        mask = Image.fromarray(mask)
                        mask.save(f"mask_values_batch{batch_idx}_sample{i}.png")
                elif k == "cond_pixel_values":
                    # Convert masked image tensor to images and save
                    for i in range(v.shape[0]):
                        img = v[i].permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        img = Image.fromarray(img)
                        img.save(f"cond_pixel_values{batch_idx}_sample{i}.png")
            else:
                print(f"{k}: {v}")
        break

    # torch.save(batch, "test_dataloader_batch_size_4_num_workers_32_prefetch_factor_2_pin_memory_True.pt")
# %%
