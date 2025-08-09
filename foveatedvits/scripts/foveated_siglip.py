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
# "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz:img",

jax.config.update("jax_platform_name", "gpu")

# img = nnx_bridge.ToNNX(
model = _siglip.Module(
                num_classes=0,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm='bfloat16',
                # posemb="rope2d_scale",   # NEW
                posemb="sincos2d",
                # unify_tiers=True, # NEW
            )
        # )


def make_uv_for_two_tier_112():
    # Coarse tier (A): 8x8 on full 224, s=28
    xs_a = (jnp.arange(8) + 0.5) * 28.0
    ys_a = (jnp.arange(8) + 0.5) * 28.0
    Xa, Ya = jnp.meshgrid(xs_a, ys_a, indexing="ij")
    xa = Xa.reshape(-1)  # [64]
    ya = Ya.reshape(-1)  # [64]
    sa = jnp.full((64,), 28.0)

    # Fovea tier (B): 8x8 in [56,168), s=14
    xs_b = 56.0 + (jnp.arange(8) + 0.5) * 14.0
    ys_b = 56.0 + (jnp.arange(8) + 0.5) * 14.0
    Xb, Yb = jnp.meshgrid(xs_b, ys_b, indexing="ij")
    xb = Xb.reshape(-1)  # [64]
    yb = Yb.reshape(-1)  # [64]
    sb = jnp.full((64,), 14.0)

    x = jnp.concatenate([xa, xb], axis=0)  # [128]
    y = jnp.concatenate([ya, yb], axis=0)  # [128]
    s = jnp.concatenate([sa, sb], axis=0)  # [128]

    u = (x / s)[None, :]  # [1,128]
    v = (y / s)[None, :]  # [1,128]
    return u, v

rng = jax.random.PRNGKey(0)
u, v = make_uv_for_two_tier_112()  # [1,128] each
variables = model.lazy_init(rng, jnp.ones((2, 112, 112, 3)), train=False, rope_uv=(u, v))
params = variables["params"]

params = load_siglip_so400m224(params)
# variables = {"params": params}


@jax.jit
def forward(p, x, rope_uv=None):                     # compiled once
    return model.apply({'params': p}, x, train=False) #, rope_uv=rope_uv)

# Load video from path
# video_path = "/home/justinyu/foveatedvits/foveatedvits/scripts/example_media/right_camera-images-rgb-soup_can_pick.mp4"
video_path = "/home/justinyu/foveatedvits/foveatedvits/scripts/example_media/top_camera-images-rgb.mp4"
video = imageio.get_reader(video_path)

# Embed video frames
real_rgb_frames_224 = []
forward_224_times_ms = []
forward_112_times_ms = []
pca_embeddings_16x16_100 = []

real_rgb_frames_112_100 = []
real_rgb_frames_112_50 = []
real_rgb_tiers_overlay = []

for frame_idx, frame in enumerate(video):
    if frame.shape[1] / frame.shape[0] > 3:
        frame = frame[:, :frame.shape[1]//2, :]
    frame_224 = resize_with_center_crop(frame, 224, 224)
    
    real_rgb_frames_224.append(frame_224)
    # start_time = time.time()
    # emb, _ = forward(params, frame_224[None]) # out: (1, 256, 1152)
    # end_time = time.time()
    # if frame_idx > 0:
    #     forward_224_times_ms.append((end_time - start_time) * 1000)
    # print(f"Time taken for SigLIP forward pass: {(end_time - start_time) * 1000} ms")
    # PCA Visualization
    import numpy as np
    from sklearn.decomposition import PCA
    
    patch_h, patch_w = 16, 16    
    # Apply PCA
    # pca = PCA(n_components=3)
    # pca.fit(np.asarray(emb).reshape(-1, 1152))
    # emb_pca = pca.transform(np.asarray(emb).reshape(-1, 1152))
    # for i in range(3):
    #     emb_pca[:, i] = (emb_pca[:, i] - emb_pca[:, i].min()) / (emb_pca[:, i].max() - emb_pca[:, i].min())
    # emb_pca = (emb_pca * 255).astype(np.uint8)
    # emb_pca = emb_pca.reshape(16, 16, 3)
    # emb_pca = cv2.resize(emb_pca, (224, 224), interpolation=cv2.INTER_NEAREST)
    # pca_embeddings_16x16_100.append(emb_pca)

    # tier one: 112x112 result at 100% center crop

    frame_112_100percent = resize_with_center_crop(frame, 112, 112, percent_center_crop=1.0)
    real_rgb_frames_112_100.append(frame_112_100percent)

    # tier one: 112x112 result at 50% center crop
    frame_112_50percent = resize_with_center_crop(frame, 112, 112, percent_center_crop=0.5)
    real_rgb_frames_112_50.append(frame_112_50percent)

    # up-resolution both to 224x224 and overlay 50 percent on the 100 percent
    frame_112_100percent_upres = cv2.resize(frame_112_100percent, (224, 224), interpolation=cv2.INTER_NEAREST)
    frame_112_100percent_upres[112//2:112*3//2, 112//2:112*3//2] = frame_112_50percent
    real_rgb_tiers_overlay.append(frame_112_100percent_upres)

    foveated_frames = np.concatenate([frame_112_100percent[None], frame_112_50percent[None]], axis=0)
    
    start_time = time.time()

    # u, v = make_uv_for_two_tier_112()  # [1,128] each
    emb, _ = forward(params, foveated_frames) # out: (2, 64, 1152)

    end_time = time.time()
    if frame_idx > 0:
        forward_112_times_ms.append((end_time - start_time) * 1000)
    print(f"Time taken for SigLIP foveated forward pass: {(end_time - start_time) * 1000} ms")

# to mp4
real_rgb_frames_224 = np.asarray(real_rgb_frames_224)
pca_embeddings_16x16_100 = np.asarray(pca_embeddings_16x16_100)
imageio.mimsave('real_rgb_frames_224.gif', real_rgb_frames_224, fps=15)
# imageio.mimsave('pca_embeddings_16x16_100.gif', pca_embeddings_16x16_100, fps=15)
# imageio.mimsave('real_rgb_frames_112_100.gif', real_rgb_frames_112_100, fps=15)
# imageio.mimsave('real_rgb_frames_112_50.gif', real_rgb_frames_112_50, fps=15)
imageio.mimsave('real_rgb_tiers_overlay.gif', real_rgb_tiers_overlay, fps=15)

# print(f"Average forward pass time for 224x224 image: {np.mean(forward_224_times_ms)} ms")
print(f"Average forward pass time for 112x112 foveated image: {np.mean(forward_112_times_ms)} ms")