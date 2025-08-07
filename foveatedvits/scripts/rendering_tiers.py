import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import foveatedvits.modeling.siglip as _siglip
import jax.numpy as jnp
import jax
import imageio
import cv2
from foveatedvits.utils.siglip_loader import load_siglip_so400m224
import time
from foveatedvits.utils.image_utils import resize_with_center_crop
from foveatedvits.utils.video_procressing_utils import resize_and_pad_video

# "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz:img",

# jax.config.update("jax_platform_name", "gpu")

# # img = nnx_bridge.ToNNX(
# model = _siglip.Module(
#                 num_classes=0,
#                 variant="So400m/14",
#                 pool_type="none",
#                 scan=True,
#                 dtype_mm='bfloat16',
#             )
#         # )

# rng = jax.random.PRNGKey(0)
# variables = model.lazy_init(rng, jnp.ones((1, 224, 224, 3)), train=False)
# params = variables["params"]

# params = load_siglip_so400m224(params)
# # variables = {"params": params}

# @jax.jit
# def forward(p, x):                     # compiled once
#     return model.apply({'params': p}, x, train=False)

# Load video from path
video_path = "/home/justinyu/foveatedvits/foveatedvits/scripts/example_media/right_camera-images-rgb-soup_can_pick.mp4"
video = imageio.get_reader(video_path)

# Original to 224x224
resize_and_pad_video(video_path, "resized_center_crop_video.mp4", target_size=224, crop_to_square=True)

# Original to 112x112
resize_and_pad_video(video_path, "resized_center_crop_video_112.mp4", target_size=112, crop_to_square=True)

# Original to 56x56
resize_and_pad_video(video_path, "resized_center_crop_video_112_50percent.mp4", target_size=112, crop_to_square=True, percent_center_crop=0.5)

# Original to 28x28
resize_and_pad_video(video_path, "resized_center_crop_video_112_25percent.mp4", target_size=112, crop_to_square=True, percent_center_crop=0.25)