# foveatedvits/utils/siglip_loader.py
import io, os, pathlib, requests, tempfile
import numpy as np
import jax.numpy as jnp
import flax
from flax.traverse_util import unflatten_dict

_SIGLIP_SO400M_224 = (
    "https://storage.googleapis.com/big_vision/"
    "siglip/webli_en_so400m_224_57633886.npz"  # gs://… –> https://storage.googleapis.com/…
)

def _cached_path(url: str, cache_dir="~/.cache/foveatedvits"):
    cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tgt = cache_dir / pathlib.Path(url).name
    if not tgt.exists():
        print(f"[siglip_loader] downloading {url} …")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False) as fp:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fp.write(chunk)
                tmp_path = pathlib.Path(fp.name)
        tmp_path.replace(tgt)
    return tgt

def _npz_to_flax_tree(npz_file, subtree="img"):
    """
    Returns a FrozenDict with the <subtree>/… slice of the checkpoint.

    The checkpoint paths look like 'params/img/embedding/kernel'.
    We keep the part *after* '/<subtree>/' and turn it into a Flax tree.
    """
    raw = np.load(npz_file, allow_pickle=False)
    wanted = {}
    needle = f"/{subtree}/"

    for full_key, value in raw.items():
        if needle not in full_key:              # skip txt tower etc.
            continue
        # keep everything after ".../<subtree>/"
        remainder = full_key.split(needle, 1)[1]        # 'embedding/kernel'
        wanted[tuple(remainder.split("/"))] = jnp.asarray(value)

    if not wanted:
        raise ValueError(f"No '{subtree}' subtree found in {npz_file}!")

    return flax.core.freeze(unflatten_dict(wanted))


def _recursive_update(dst, src):
    """Overwrite leaves in `dst` with those in `src`, skipping extra keys."""
    for k, v in src.items():
        if k not in dst:
            # checkpoint has a param the model doesn't use → skip it
            continue
        if isinstance(v, (dict, flax.core.FrozenDict)):
            dst[k] = _recursive_update(dict(dst[k]), dict(v))
        else:                      # leaf tensor
            dst[k] = v
    return dst

def load_siglip_so400m224(params, cache_dir="~/.cache/foveatedvits"):
    """Return a new FrozenDict with SigLIP-So400m/14 (224 px) weights installed."""
    ckpt_path = _cached_path(_SIGLIP_SO400M_224, cache_dir)
    siglip_tree = _npz_to_flax_tree(ckpt_path, subtree="img")

    # sanity-check a single leaf to fail early if shapes diverge
    ref = params["embedding"]["kernel"]
    if siglip_tree["embedding"]["kernel"].shape != ref.shape:
        raise ValueError("Checkpoint does not match model shape/version.")

    params_f = flax.core.unfreeze(params)
    params_f = _recursive_update(params_f, dict(siglip_tree))
    return flax.core.freeze(params_f)
