"""
ncde/model.py
=============
Neural CDE models for ADNI ADAS13 prediction using JAX + Equinox + Diffrax.

Two model variants:
  1. BaselineNCDE     – clinical time series → NCDE → predict ADAS13
  2. MultimodalNCDE   – imaging encoder fuses with clinical features → NCDE → predict

Architecture overview (for MultimodalNCDE):
  ┌─────────────┐
  │ Clinical     │──┐
  │ Features (6) │  │   ┌──────────────┐     ┌───────┐     ┌──────────┐
  └─────────────┘  ├──▶│ Augmented     │────▶│ NCDE  │────▶│ Readout  │──▶ ADAS13
  ┌─────────────┐  │   │ Control Path  │     │ (ODE) │     │ (Linear) │
  │ Imaging     │──┤   └──────────────┘     └───────┘     └──────────┘
  │ Encoder MLP │  │
  └─────────────┘  │
                   │
  (Concatenated at each time step)

The NCDE integrates a controlled differential equation:
  dh/dt = f_θ(h) · dX/dt
where X is the (interpolated) control path and f_θ is a learned vector field.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax


# ═══════════════════════════════════════════════════════════════════════════
# Vector Field: the core learned function for the CDE
# ═══════════════════════════════════════════════════════════════════════════

class CDEVectorField(eqx.Module):
    """
    The vector field f_θ(h) for the Neural CDE.

    Given hidden state h ∈ R^d, produces a matrix M ∈ R^(d × input_dim).
    The CDE dynamics are: dh/dt = M(h) · dX/dt.

    Architecture: h → Linear → GeLU → Linear → GeLU → Linear → reshape to (d, input_dim)
    """
    layers: list
    _hidden_dim: int = eqx.field(static=True)
    _input_dim: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int, input_dim: int, width: int = 128, *, key: jax.Array):
        """
        Args:
            hidden_dim: dimension of the hidden state h
            input_dim: dimension of the control signal X (= augmented feature dim)
            width: width of hidden layers in the vector field MLP
            key: PRNG key for initialization
        """
        keys = jax.random.split(key, 3)

        self._hidden_dim = hidden_dim
        self._input_dim = input_dim

        # MLP: hidden_dim → width → width → hidden_dim * input_dim
        self.layers = [
            eqx.nn.Linear(hidden_dim, width, key=keys[0]),
            eqx.nn.Linear(width, width, key=keys[1]),
            eqx.nn.Linear(width, hidden_dim * input_dim, key=keys[2]),
        ]

    def __call__(self, h: jax.Array) -> jax.Array:
        """
        Compute the vector field output.

        Args:
            h: hidden state (hidden_dim,)

        Returns:
            Matrix of shape (hidden_dim, input_dim)
        """
        x = h
        # Layer 1: linear + GeLU
        x = jax.nn.gelu(self.layers[0](x))
        # Layer 2: linear + GeLU
        x = jax.nn.gelu(self.layers[1](x))
        # Layer 3: linear (no activation, produces the matrix)
        x = self.layers[2](x)
        # Reshape to (hidden_dim, input_dim) and apply tanh for stability
        return jnp.tanh(x.reshape(self._hidden_dim, self._input_dim))


# ═══════════════════════════════════════════════════════════════════════════
# CDE Function wrapper for diffrax
# ═══════════════════════════════════════════════════════════════════════════

class CDEFunc(eqx.Module):
    """
    Wraps the vector field + control path derivative for use with diffrax.

    The CDE: dh/dt = f_θ(h) · dX/dt

    We pass dX/dt via the `args` mechanism. The control path derivative is
    computed by the caller at each solver step via linear interpolation.
    """
    vector_field: CDEVectorField

    def __call__(self, t: float, h: jax.Array, args: jax.Array) -> jax.Array:
        """
        Compute dh/dt = f_θ(h) · dX/dt.

        Args:
            t: current time (unused directly, but required by diffrax)
            h: hidden state (hidden_dim,)
            args: dX/dt — the control path derivative at time t, shape (input_dim,)

        Returns:
            dh/dt of shape (hidden_dim,)
        """
        # f_θ(h) has shape (hidden_dim, input_dim)
        matrix = self.vector_field(h)
        # dX/dt has shape (input_dim,)
        dxdt = args
        # Matrix-vector product: (hidden_dim, input_dim) @ (input_dim,) = (hidden_dim,)
        return matrix @ dxdt


# ═══════════════════════════════════════════════════════════════════════════
# Imaging Feature Encoder (MLP acting as CNN analogue)
# ═══════════════════════════════════════════════════════════════════════════

class ImagingEncoder(eqx.Module):
    """
    Encodes imaging-related features into a learned embedding.

    In the current setup (no raw images), this takes the imaging indicator
    features (has_mri, has_fdg_pet) and diagnosis features and produces a
    dense embedding. When raw images are available, this can be replaced
    by a CNN.

    Architecture: input_dim → 32 → GeLU → embed_dim → GeLU
    """
    net: eqx.nn.MLP

    def __init__(self, input_dim: int, embed_dim: int = 32, *, key: jax.Array):
        """
        Args:
            input_dim: number of imaging-related input features
            embed_dim: output embedding dimension
            key: PRNG key
        """
        self.net = eqx.nn.MLP(
            in_size=input_dim,
            out_size=embed_dim,
            width_size=32,
            depth=2,
            activation=jax.nn.gelu,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Encode imaging features at one time step.

        Args:
            x: imaging features (input_dim,)

        Returns:
            Embedding vector (embed_dim,)
        """
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════
# Readout Head
# ═══════════════════════════════════════════════════════════════════════════

class ReadoutHead(eqx.Module):
    """
    Maps the CDE hidden state to ADAS13 prediction.
    Architecture: hidden_dim → 32 → ReLU → 1
    """
    net: eqx.nn.MLP

    def __init__(self, hidden_dim: int, *, key: jax.Array):
        self.net = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=1,
            width_size=32,
            depth=1,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, h: jax.Array) -> jax.Array:
        """Map hidden state to scalar prediction."""
        return self.net(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Model 1: Baseline NCDE (clinical features only)
# ═══════════════════════════════════════════════════════════════════════════

class BaselineNCDE(eqx.Module):
    """
    Neural CDE using only clinical time series features.

    1. Project raw features to hidden dim (initial state)
    2. Build piecewise-linear control path from features
    3. Integrate the CDE: dh/dt = f_θ(h) · dX/dt
    4. Read out ADAS13 prediction from hidden states
    """
    input_proj: eqx.nn.Linear   # project features → hidden dim (initial state)
    cde_func: CDEFunc            # the CDE dynamics
    readout: ReadoutHead         # hidden → ADAS13

    hidden_dim: int = eqx.field(static=True)  # static metadata
    feature_dim: int = eqx.field(static=True)

    def __init__(
        self,
        feature_dim: int = 6,
        hidden_dim: int = 64,
        vf_width: int = 128,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 3)

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # The control path dimension is feature_dim + 1 (we prepend time as a channel)
        control_dim = feature_dim + 1

        self.input_proj = eqx.nn.Linear(feature_dim, hidden_dim, key=keys[0])
        self.cde_func = CDEFunc(
            vector_field=CDEVectorField(hidden_dim, control_dim, width=vf_width, key=keys[1])
        )
        self.readout = ReadoutHead(hidden_dim, key=keys[2])

    def __call__(
        self,
        time: jax.Array,      # (T,) normalized time
        features: jax.Array,  # (T, F) normalized features
        mask: jax.Array,      # (T, F) observation mask
        length: jax.Array,    # scalar: actual sequence length
    ) -> jax.Array:
        """
        Forward pass for a single subject.

        Args:
            time: (T,) time points
            features: (T, F) feature values
            mask: (T, F) observation mask
            length: scalar, actual sequence length

        Returns:
            predictions: (T,) ADAS13 predictions at each time step
        """
        seq_len = time.shape[0]

        # Build augmented control path: prepend time as extra channel
        # X(t) = [t, features(t)] ∈ R^(F+1)
        control_path = jnp.concatenate([time[:, None], features], axis=1)  # (T, F+1)

        # Initial hidden state from first observation
        h0 = self.input_proj(features[0])  # (hidden_dim,)

        # --- Integrate the CDE using Euler steps (fast, stable for our data) ---
        # We use a simple scan over time steps instead of diffrax for efficiency
        # with variable-length padded sequences

        def _step(carry, idx):
            """One Euler step of the CDE."""
            h_prev = carry

            # Compute dX/dt via finite differences
            # At step i: dX/dt ≈ (X[i+1] - X[i]) / (t[i+1] - t[i])
            # Clamp idx to avoid out-of-bounds
            idx_next = jnp.minimum(idx + 1, seq_len - 1)
            dt = jnp.maximum(time[idx_next] - time[idx], 1e-6)
            dxdt = (control_path[idx_next] - control_path[idx]) / dt

            # CDE step: h_{i+1} = h_i + f_θ(h_i) · dX/dt · dt
            dh = self.cde_func(time[idx], h_prev, dxdt) * dt

            # Mask: only update if we're within the actual sequence
            active = (idx < length).astype(jnp.float32)
            h_new = h_prev + dh * active

            return h_new, h_new

        # Scan over all time steps
        indices = jnp.arange(seq_len)
        _, all_hidden = jax.lax.scan(_step, h0, indices)  # all_hidden: (T, hidden_dim)

        # Read out predictions at each time step
        predictions = jax.vmap(self.readout)(all_hidden)  # (T,)

        return predictions


# ═══════════════════════════════════════════════════════════════════════════
# Model 2: Multimodal NCDE (clinical + imaging encoder)
# ═══════════════════════════════════════════════════════════════════════════

class MultimodalNCDE(eqx.Module):
    """
    Multimodal Neural CDE with end-to-end imaging feature encoder.

    1. Split features into clinical and imaging subsets
    2. Pass imaging features through encoder MLP at each time step
    3. Concatenate clinical + imaging embeddings → augmented control path
    4. Integrate CDE with augmented path
    5. Read out predictions

    The imaging encoder trains jointly with the CDE — fully end-to-end.
    """
    imaging_encoder: ImagingEncoder
    input_proj: eqx.nn.Linear
    cde_func: CDEFunc
    readout: ReadoutHead

    hidden_dim: int = eqx.field(static=True)
    feature_dim: int = eqx.field(static=True)
    img_feat_dim: int = eqx.field(static=True)
    clinical_feat_dim: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    # Feature indices for splitting (static)
    # In our data: [ADAS13, TOTSCORE, DIAGNOSIS, DXNORM, has_mri, has_fdg_pet]
    # Clinical: ADAS13(0), TOTSCORE(1)
    # Imaging-related: DIAGNOSIS(2), DXNORM(3), has_mri(4), has_fdg_pet(5)
    _clinical_idx: tuple = eqx.field(static=True)
    _imaging_idx: tuple = eqx.field(static=True)

    def __init__(
        self,
        feature_dim: int = 6,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        vf_width: int = 128,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 4)

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Split features: clinical (first 2) vs imaging-related (last 4)
        self._clinical_idx = (0, 1)      # ADAS13, TOTSCORE
        self._imaging_idx = (2, 3, 4, 5) # DIAGNOSIS, DXNORM, has_mri, has_fdg_pet

        self.clinical_feat_dim = len(self._clinical_idx)
        self.img_feat_dim = len(self._imaging_idx)

        # Imaging encoder: 4 features → embed_dim embedding
        self.imaging_encoder = ImagingEncoder(
            input_dim=self.img_feat_dim,
            embed_dim=embed_dim,
            key=keys[0],
        )

        # Augmented control path dim: time(1) + clinical(2) + embedding(embed_dim)
        augmented_dim = 1 + self.clinical_feat_dim + embed_dim

        # Initial state projection from full feature set
        self.input_proj = eqx.nn.Linear(feature_dim, hidden_dim, key=keys[1])

        # CDE with augmented control path
        self.cde_func = CDEFunc(
            vector_field=CDEVectorField(hidden_dim, augmented_dim, width=vf_width, key=keys[2])
        )
        self.readout = ReadoutHead(hidden_dim, key=keys[3])

    def _encode_step(self, features_t: jax.Array) -> jax.Array:
        """
        Encode features at a single time step into augmented features.

        Args:
            features_t: (F,) features at one time step

        Returns:
            augmented: (clinical_dim + embed_dim,) augmented features
        """
        # Extract clinical and imaging features
        clinical = features_t[jnp.array(self._clinical_idx)]
        imaging = features_t[jnp.array(self._imaging_idx)]

        # Encode imaging → embedding
        img_embed = self.imaging_encoder(imaging)  # (embed_dim,)

        # Concatenate clinical + imaging embedding
        return jnp.concatenate([clinical, img_embed])

    def __call__(
        self,
        time: jax.Array,
        features: jax.Array,
        mask: jax.Array,
        length: jax.Array,
    ) -> jax.Array:
        """
        Forward pass for a single subject (multimodal).

        Args:
            time: (T,) time points
            features: (T, F) raw features
            mask: (T, F) observation mask
            length: scalar, actual sequence length

        Returns:
            predictions: (T,) ADAS13 predictions
        """
        seq_len = time.shape[0]

        # Encode features at each time step (imaging encoder applied at each step)
        augmented_features = jax.vmap(self._encode_step)(features)  # (T, clinical+embed)

        # Build control path: [time, augmented_features]
        control_path = jnp.concatenate([time[:, None], augmented_features], axis=1)

        # Initial hidden state from raw features
        h0 = self.input_proj(features[0])

        # CDE integration via scan
        def _step(carry, idx):
            h_prev = carry

            idx_next = jnp.minimum(idx + 1, seq_len - 1)
            dt = jnp.maximum(time[idx_next] - time[idx], 1e-6)
            dxdt = (control_path[idx_next] - control_path[idx]) / dt

            dh = self.cde_func(time[idx], h_prev, dxdt) * dt

            active = (idx < length).astype(jnp.float32)
            h_new = h_prev + dh * active

            return h_new, h_new

        indices = jnp.arange(seq_len)
        _, all_hidden = jax.lax.scan(_step, h0, indices)

        predictions = jax.vmap(self.readout)(all_hidden)

        return predictions


# ═══════════════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════════════

def create_model(
    model_type: str = "baseline",
    feature_dim: int = 6,
    hidden_dim: int = 64,
    embed_dim: int = 32,
    vf_width: int = 128,
    *,
    key: jax.Array,
) -> eqx.Module:
    """
    Create a Neural CDE model.

    Args:
        model_type: "baseline" or "multimodal"
        feature_dim: number of input features per time step
        hidden_dim: CDE hidden state dimension
        embed_dim: imaging encoder output dimension (multimodal only)
        vf_width: vector field MLP width
        key: PRNG key

    Returns:
        An Equinox module
    """
    if model_type == "baseline":
        return BaselineNCDE(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            vf_width=vf_width,
            key=key,
        )
    elif model_type == "multimodal":
        return MultimodalNCDE(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            vf_width=vf_width,
            key=key,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'baseline' or 'multimodal'.")
