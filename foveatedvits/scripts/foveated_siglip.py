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
            )
        # )

rng = jax.random.PRNGKey(0)
variables = model.lazy_init(rng, jnp.ones((1, 224, 224, 3)), train=False)
params = variables["params"]

params = load_siglip_so400m224(params)
# variables = {"params": params}

@jax.jit
def forward(p, x):                     # compiled once
    return model.apply({'params': p}, x, train=False)

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
    start_time = time.time()
    emb, _ = forward(params, frame_224[None]) # out: (1, 256, 1152)
    end_time = time.time()
    if frame_idx > 0:
        forward_224_times_ms.append((end_time - start_time) * 1000)
    print(f"Time taken for SigLIP forward pass: {(end_time - start_time) * 1000} ms")
    # PCA Visualization
    import numpy as np
    from sklearn.decomposition import PCA
    
    patch_h, patch_w = 16, 16    
    # Apply PCA
    pca = PCA(n_components=3)
    pca.fit(np.asarray(emb).reshape(-1, 1152))
    emb_pca = pca.transform(np.asarray(emb).reshape(-1, 1152))
    for i in range(3):
        emb_pca[:, i] = (emb_pca[:, i] - emb_pca[:, i].min()) / (emb_pca[:, i].max() - emb_pca[:, i].min())
    emb_pca = (emb_pca * 255).astype(np.uint8)
    emb_pca = emb_pca.reshape(16, 16, 3)
    emb_pca = cv2.resize(emb_pca, (224, 224), interpolation=cv2.INTER_NEAREST)
    pca_embeddings_16x16_100.append(emb_pca)

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
    emb, _ = forward(params, foveated_frames) # out: (2, 64, 1152)

    # import pdb; pdb.set_trace()
    end_time = time.time()
    if frame_idx > 0:
        forward_112_times_ms.append((end_time - start_time) * 1000)
    print(f"Time taken for SigLIP foveated forward pass: {(end_time - start_time) * 1000} ms")

# to mp4
real_rgb_frames_224 = np.asarray(real_rgb_frames_224)
pca_embeddings_16x16_100 = np.asarray(pca_embeddings_16x16_100)
imageio.mimsave('real_rgb_frames_224.gif', real_rgb_frames_224, fps=15)
imageio.mimsave('pca_embeddings_16x16_100.gif', pca_embeddings_16x16_100, fps=15)
# imageio.mimsave('real_rgb_frames_112_100.gif', real_rgb_frames_112_100, fps=15)
# imageio.mimsave('real_rgb_frames_112_50.gif', real_rgb_frames_112_50, fps=15)
imageio.mimsave('real_rgb_tiers_overlay.gif', real_rgb_tiers_overlay, fps=15)

print(f"Average forward pass time for 224x224 image: {np.mean(forward_224_times_ms)} ms")
print(f"Average forward pass time for 112x112 foveated image: {np.mean(forward_112_times_ms)} ms")