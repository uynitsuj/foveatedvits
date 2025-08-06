import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import foveatedvits.modeling.siglip as _siglip
import jax.numpy as jnp
import jax
import imageio
from foveatedvits.utils.siglip_loader import load_siglip_so400m224

# "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz:img",

# img = nnx_bridge.ToNNX(
model = _siglip.Module(
                num_classes=0,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm='bfloat16',
            )
        # )

rng = jax.random.PRNGKey(0)
variables = model.lazy_init(rng, jnp.ones((1, 224, 224, 3)), train=False)
params = variables["params"]

params = load_siglip_so400m224(params)
variables = {"params": params}

# Load video from path
video_path = "/home/justinyu/foveatedvits/foveatedvits/scripts/example_media/episode_000000.mp4"
video = imageio.get_reader(video_path)

# Embed video frames
for frame in video:
    frame = jax.image.resize(
        jnp.asarray(frame, jnp.float32) / 255.0, (224, 224, 3), "bilinear"
    )
    emb, _ = model.apply(variables, frame[None], train=False)
    import pdb; pdb.set_trace()
