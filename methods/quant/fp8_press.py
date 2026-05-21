from dataclasses import dataclass
import logging
from typing import Any

import torch
from torch import Tensor

from methods.quant.quant_press import QuantizedPress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FP8 format constants
# ---------------------------------------------------------------------------

# E4M3: 1 sign, 4 exponent, 3 mantissa (bias=7)
# Max normal: ±448.0  (E=14,M=7 → 2^7 × 1.875)
# No infinities – exponent 15 encodes NaN
_E4M3_EXPONENT_BITS = 4
_E4M3_MANTISSA_BITS = 3
_E4M3_BIAS = 7
_E4M3_MAX_NORMAL = 448.0
_E4M3_MAX_EXP = 15  # encoded exponent max for normal values
_E4M3_MANT_SCALE = 8  # 2^3 = 8 discrete mantissa levels

# E5M2: 1 sign, 5 exponent, 2 mantissa (bias=15)
# Max normal: ±57344.0
# Exponent 31 → inf (mant=0) or NaN (mant!=0)
_E5M2_EXPONENT_BITS = 5
_E5M2_MANTISSA_BITS = 2
_E5M2_BIAS = 15
_E5M2_MAX_NORMAL = 57344.0
_E5M2_MAX_EXP = 30
_E5M2_MANT_SCALE = 4  # 2^2 = 4 discrete mantissa levels


# ===========================================================================
# Core FP8 quantise / dequantise
# ===========================================================================

def _quantise_e4m3(x_f32: Tensor, saturation_mode:str = 'SAT', rounding_mode:str = 'NEAREST_EVEN') -> Tensor:
    """
    Quantise a float32 tensor to e4m3 fp8, returned as uint8 bit patterns.

    E4M3 layout (MSB → LSB): [s][eeee][mmm]

    Strategy (per element)
    ----------------------
    1. Specials (NaN / Inf) → NaN (0b01111111, 0b11111111) ; all E and M bits are 1
    2. Zeros → ±0
    3. finfo.largest ≤ |x| → clamp to MAX_NORMAL
    4. |x| < 2⁻⁹ (half the smallest subnormal) → flush to zero
    5. Otherwise: compute exponent + rounded mantissa; carry to exponent
       on mantissa overflow; saturate exponent when needed.
    """
    sign = x_f32 < 0  # bool

    # saturate all overflowing values to max e4m3 value
    a = torch.abs(x_f32)
    a = a.clamp(max=_E4M3_MAX_NORMAL)

    # --- Specials ---
    is_nan = torch.isnan(x_f32)
    is_inf = torch.isinf(x_f32)
    a = torch.where(is_inf, torch.clamp(a, max=_E4M3_MAX_NORMAL), a)
    is_special = is_nan 

    # flush to zero below 2⁻⁹ as this is smallest possible e4m3 subnormal value
    is_zero = (a == 0) | (a < (1.0 / 512.0))  

    # --- Normal / subnormal quantisation ---
    mant_f32, exp_f32 = torch.frexp(a.clamp(min=torch.finfo(torch.float32).tiny))

    # frexp returns: a = mant * 2**exp  with  mant ∈ [0.5, 1)
    # floor(log2(a)) = exp - 1  (since mant ∈ [0.5, 1) means log2(mant) ∈ [-1, 0))

    binary_exp = exp_f32 - 1  # floor(log2(a))
    scaled_mant = mant_f32 * 2.0  # now ∈ [1, 2)

    # Encoded exponent (bias 7)
    enc_exp = binary_exp + _E4M3_BIAS  # may be negative for subnormals

    # Rounded mantissa (3 bits → 0 … 7)
    mant_frac = scaled_mant - 1  # ∈ [0, 1)
    mant_int = torch.round(mant_frac * _E4M3_MANT_SCALE).to(torch.int32)

    # ----- Carry from mantissa round-to-8 -----
    carry = mant_int >= _E4M3_MANT_SCALE
    enc_exp = torch.where(carry, enc_exp + 1, enc_exp)
    mant_int = torch.where(carry, torch.zeros_like(mant_int), mant_int)

    # ----- Subnormal path (enc_exp ≤ 0) -----
    # For subnormals: value = m / 8 × 2⁻⁶
    # We need: mant_scaled × 2^(binary_exp) = (m/8) × 2⁻⁶
    # → m = mant_scaled × 2^(binary_exp + 6) × 8
    subnormal = enc_exp <= 0
    subnorm_mant_frac = torch.where(
        subnormal,
        scaled_mant * (2.0 ** (binary_exp + 6).float()),
        torch.zeros_like(scaled_mant),
    )
    subnorm_mant_int = torch.round(subnorm_mant_frac * _E4M3_MANT_SCALE).to(
        torch.int32
    )
    subnorm_mant_int = subnorm_mant_int.clamp(0, _E4M3_MANT_SCALE - 1)

    # override if subnormal
    mant_int = torch.where(subnormal, subnorm_mant_int, mant_int)
    enc_exp = torch.where(subnormal, torch.zeros_like(enc_exp), enc_exp)

    # ----- Saturate exponent overflow -----
    overflow = enc_exp > _E4M3_MAX_EXP
    enc_exp = torch.where(overflow, torch.tensor(_E4M3_MAX_EXP, dtype=torch.int32), enc_exp) # clamp to 15
    mant_int = torch.where(
        overflow,
        torch.full_like(mant_int, _E4M3_MANT_SCALE - 1),
        mant_int,
    )

    # ----- Flush to zero if lesser than lowest subnormal value -----
    mant_int = torch.where(is_zero, torch.zeros_like(mant_int), mant_int)
    enc_exp = torch.where(is_zero, torch.zeros_like(enc_exp), enc_exp)

    # ----- Assemble fp8 byte -----
    sign_bits = sign.to(torch.int32) << 7
    exp_bits = enc_exp.to(torch.int32) << _E4M3_MANTISSA_BITS
    mant_bits = mant_int.to(torch.int32)
    packed = (sign_bits | exp_bits | mant_bits).to(torch.uint8)

    # ----- Overwrite specials with NaN -----
    nan_byte = torch.tensor(0x7f, dtype=torch.uint8, device=x_f32.device)
    # negative NaN: 0b11111111 = 0xff
    nan_byte_neg = torch.tensor(0xff, dtype=torch.uint8, device=x_f32.device)
    packed = torch.where(is_special & ~sign, nan_byte, packed)
    packed = torch.where(is_special & sign, nan_byte_neg, packed)

    return packed


def _dequantise_e4m3(packed: Tensor, shape: tuple, orig_dtype: torch.dtype) -> Tensor:
    """Decode uint8 fp8 e4m3 patterns back to the original floating-point dtype."""
    p = packed.to(torch.int32)
    sign = ((p >> 7) & 1).to(torch.float32) * -2 + 1  # 0→1, 1→-1
    enc_exp = (p >> _E4M3_MANTISSA_BITS) & 0xF
    mant = p & (_E4M3_MANT_SCALE - 1)

    mant_frac = mant.to(torch.float32) / float(_E4M3_MANT_SCALE)

    # Normal path  (enc_exp >= 1)
    normal = enc_exp >= 1
    normal_val = (1.0 + mant_frac) * (2.0 ** (enc_exp.float() - _E4M3_BIAS))

    # Subnormal path  (enc_exp == 0)
    subnormal = enc_exp == 0
    subnormal_val = mant_frac * (2.0 ** (-_E4M3_BIAS + 1))

    # NaN → NaN
    is_nan = (enc_exp == (_E4M3_MAX_EXP)) & (mant > 0)
    is_nan = is_nan.to(torch.bool)

    val = torch.where(normal, normal_val, subnormal_val)
    val = torch.where(is_nan, torch.tensor(float("nan"), dtype=torch.float32, device=packed.device), val)
    val = val * sign

    return val.reshape(shape).to(orig_dtype)


# ---------------------------------------------------------------------------
# E5M2
# ---------------------------------------------------------------------------

def _quantise_e5m2(x_f32: Tensor, **kwargs) -> Tensor:
    """
    Quantise a float32 tensor to e5m2 fp8, returned as uint8 bit patterns.

    E5M2 layout (MSB → LSB): [s][eeeee][mm]
    """
    sign = x_f32 < 0
    a = torch.abs(x_f32)
    a = a.clamp(max=_E5M2_MAX_NORMAL)

    is_nan = torch.isnan(x_f32)
    is_inf = torch.isinf(x_f32)
    is_zero = (a == 0) | (a < (2.0 ** (-16)))  # flush to zero

    mant_f32, exp_f32 = torch.frexp(a.clamp(min=torch.finfo(torch.float32).tiny))
    binary_exp = exp_f32 - 1
    scaled_mant = mant_f32 * 2.0

    enc_exp = binary_exp + _E5M2_BIAS
    mant_frac = scaled_mant - 1
    mant_int = torch.round(mant_frac * _E5M2_MANT_SCALE).to(torch.int32)

    carry = mant_int >= _E5M2_MANT_SCALE
    enc_exp = torch.where(carry, enc_exp + 1, enc_exp)
    mant_int = torch.where(carry, torch.zeros_like(mant_int), mant_int)

    # Subnormal
    subnormal = enc_exp <= 0
    subnorm_mant_frac = torch.where(
        subnormal,
        scaled_mant * (2.0 ** (binary_exp + _E5M2_BIAS - 1).float()),
        torch.zeros_like(scaled_mant),
    )
    subnorm_mant_int = torch.round(subnorm_mant_frac * _E5M2_MANT_SCALE).to(torch.int32)
    subnorm_mant_int = subnorm_mant_int.clamp(0, _E5M2_MANT_SCALE - 1)
    mant_int = torch.where(subnormal, subnorm_mant_int, mant_int)
    enc_exp = torch.where(subnormal, torch.zeros_like(enc_exp), enc_exp)

    # Saturate exponent overflow → max (but keep inf path separate)
    overflow = enc_exp > _E5M2_MAX_EXP
    enc_exp = torch.where(overflow, torch.tensor(_E5M2_MAX_EXP, dtype=torch.int32), enc_exp)
    mant_int = torch.where(
        overflow,
        torch.full_like(mant_int, _E5M2_MANT_SCALE - 1),
        mant_int,
    )

    mant_int = torch.where(is_zero, torch.zeros_like(mant_int), mant_int)
    enc_exp = torch.where(is_zero, torch.zeros_like(enc_exp), enc_exp)

    sign_bits = sign.to(torch.int32) << 7
    exp_bits = enc_exp.to(torch.int32) << _E5M2_MANTISSA_BITS
    mant_bits = mant_int.to(torch.int32)
    packed = (sign_bits | exp_bits | mant_bits).to(torch.uint8)

    # Inf / NaN →  exponent all-ones
    inf_exp = _E5M2_MAX_EXP + 1  # 31
    # positive inf: 0b01111100 = 0x7c
    inf_byte = torch.tensor(
        (inf_exp << _E5M2_MANTISSA_BITS) & 0xFF,
        dtype=torch.uint8,
        device=x_f32.device,
    )
    # negative inf: 0b11111100 = 0xfc
    inf_byte_neg = torch.tensor(0x80 | inf_byte.item(), dtype=torch.uint8, device=x_f32.device)
    # NaN: any all-ones-exp with non-zero mantissa
    nan_byte = torch.tensor(
        (inf_exp << _E5M2_MANTISSA_BITS) | 1,
        dtype=torch.uint8,
        device=x_f32.device,
    )
    nan_byte_neg = torch.tensor(0x80 | nan_byte.item(), dtype=torch.uint8, device=x_f32.device)

    is_pos_inf = is_inf & ~sign
    is_neg_inf = is_inf & sign
    is_pos_nan = is_nan & ~sign
    is_neg_nan = is_nan & sign

    packed = torch.where(is_pos_inf, inf_byte, packed)
    packed = torch.where(is_neg_inf, inf_byte_neg, packed)
    packed = torch.where(is_pos_nan, nan_byte, packed)
    packed = torch.where(is_neg_nan, nan_byte_neg, packed)

    return packed


def _dequantise_e5m2(packed: Tensor, shape: tuple, orig_dtype: torch.dtype) -> Tensor:
    """Decode uint8 fp8 e5m2 patterns back to the original floating-point dtype."""
    p = packed.to(torch.int32)
    sign = ((p >> 7) & 1).to(torch.float32) * -2 + 1
    enc_exp = (p >> _E5M2_MANTISSA_BITS) & 0x1F
    mant = p & (_E5M2_MANT_SCALE - 1)

    mant_frac = mant.to(torch.float32) / float(_E5M2_MANT_SCALE)
    all_ones = _E5M2_MAX_EXP + 1  # 31

    normal = (enc_exp >= 1) & (enc_exp < all_ones)
    normal_val = (1.0 + mant_frac) * (2.0 ** (enc_exp.float() - _E5M2_BIAS))

    subnormal = enc_exp == 0
    subnormal_val = mant_frac * (2.0 ** (-_E5M2_BIAS + 1))

    is_nan = (enc_exp == all_ones) & (mant > 0)
    is_inf = (enc_exp == all_ones) & (mant == 0)

    val = torch.where(normal, normal_val, subnormal_val)
    val = torch.where(
        is_nan,
        torch.tensor(float("nan"), dtype=torch.float32, device=packed.device),
        val,
    )
    val = torch.where(
        is_inf,
        torch.tensor(float("inf"), dtype=torch.float32, device=packed.device),
        val,
    )
    val = val * sign

    return val.reshape(shape).to(orig_dtype)


# ===========================================================================
# Dispatch map
# ===========================================================================

_QUANTISE_MAP = {
    "e4m3": _quantise_e4m3,
    "e5m2": _quantise_e5m2,
}

_DEQUANTISE_MAP = {
    "e4m3": _dequantise_e4m3,
    "e5m2": _dequantise_e5m2,
}


# ===========================================================================
# FP8Press
# ===========================================================================


@dataclass
class FP8Press(QuantizedPress):
    """
    BF16 → FP8 quantisation for KV-cache compression.

    Converts each key / value tensor from its original floating-point type
    (typically bfloat16 or float16) to an 8-bit floating-point format, then
    back again on decode.  This yields a **2x compression ratio** (16 bpc → 8
    bpc) with the rounding behaviour of IEEE-style floating point.

    Parameters
    ----------
    format : str, either ``"e4m3"`` (default) or ``"e5m2"``.
        ``e4m3`` - 4 exponent bits, 3 mantissa bits.  Dynamic range ±240
        with finer precision for values near zero.  This is the default
        because KV-cache values rarely exceed ±100 (see instrumented outputs).
        ``e5m2`` - 5 exponent bits, 2 mantissa bits.  Much larger dynamic
        range (±57344) but coarser precision.  Prefer this when the KV tensors
        contain large-magnitude values.

    Notes
    -----
    * The compressed representation is a raw ``uint8`` tensor (1 byte per
      element) plus a metadata dictionary with shape and dtype information.
    * The default format (e4m3) matches `torch.float8_e4m3fn`.
    * Works on CPU and GPU; no special hardware required.
    """

    format: str = "e4m3"
    rounding_mode:str = "STOCHASTIC" # nearest, stochastic
    saturation_mode:str = "SAT" # sat, unsat

    def __post_init__(self) -> None:
        self.format = self.format.lower()
        if self.format not in _QUANTISE_MAP:
            raise ValueError(
                f"Unknown FP8 format '{self.format}'. "
                f"Supported: {list(_QUANTISE_MAP.keys())}"
            )
        super().__post_init__()

    def _encode(self, x: Tensor) -> tuple[Any, dict]:
        quantise_fn = _QUANTISE_MAP[self.format]
        x_f32 = x.detach().float()
        compressed = quantise_fn(x_f32, saturation_mode=self.saturation_mode, rounding_mode=self.rounding_mode)
        meta = {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "fp8_format": self.format,
        }
        return compressed, meta

    def _decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        dequantise_fn = _DEQUANTISE_MAP[meta["fp8_format"]]
        orig_dtype_str = meta["dtype"]
        orig_dtype = getattr(torch, orig_dtype_str.split(".")[-1], torch.bfloat16)
        return dequantise_fn(compressed, original_shape, orig_dtype)


# ===========================================================================
# NativeFP8Press  (uses PyTorch's built-in float8 dtypes)
# ===========================================================================

_FP8_DTYPE_MAP = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}


@dataclass
class NativeFP8Press(QuantizedPress):
    """
    BF16 → FP8 quantisation using PyTorch's native ``float8`` dtypes.

    Unlike `FP8Press`, this class delegates entirely to PyTorch's built-in
    float8 conversion kernels (``torch.float8_e4m3fn`` / ``torch.float8_e5m2``).
    The compressed representation is stored directly as a ``torch.float8``
    tensor — no manual bit manipulation.

    This yields a **2x compression ratio** (16 bpc → 8 bpc).

    Parameters
    ----------
    format : str, either ``"e4m3"`` (default) or ``"e5m2"``.
    """

    format: str = "e4m3"
    saturation_mode:str = "SAT"
    rounding_mode:str = "STOCHASTIC"

    def __post_init__(self) -> None:
        self.format = self.format.lower()
        if self.format not in _FP8_DTYPE_MAP:
            raise ValueError(
                f"Unknown FP8 format '{self.format}'. "
                f"Supported: {list(_FP8_DTYPE_MAP.keys())}"
            )
        self._fp8_dtype = _FP8_DTYPE_MAP[self.format]
        super().__post_init__()

    def _encode(self, x: Tensor) -> tuple[Any, dict]:
        compressed = x.detach().to(self._fp8_dtype)
        meta = {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "fp8_format": self.format,
        }
        return compressed, meta

    def _decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        orig_dtype_str = meta["dtype"]
        orig_dtype = getattr(torch, orig_dtype_str.split(".")[-1], torch.bfloat16)
        return compressed.to(torch.float32).to(orig_dtype).reshape(original_shape)

    @staticmethod
    def _measure_bytes(compressed: Any, meta: dict) -> int:
        if isinstance(compressed, torch.Tensor):
            return compressed.numel() * compressed.element_size()
        return super(NativeFP8Press, NativeFP8Press)._measure_bytes(compressed, meta)
