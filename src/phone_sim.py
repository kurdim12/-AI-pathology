"""
Phone-capture degradation simulator (deterministic, for *evaluation*).

``src/data.py`` applies *random* augmentations during training to teach the
model phone-robustness. This module is the complementary *measurement* tool: it
applies **controlled, parameterised** degradations to a clean image so we can
quantify how much the model's triage decisions hold up as conditions worsen
(see ``src/robustness.py``).

Each knob maps to a real artefact of photographing a microscope eyepiece with a
phone:

    brightness   uneven illumination / exposure
    blur         defocus or hand motion
    jpeg         phone camera compression artefacts
    rotate       the slide/phone not being square to the field
    color_shift  white-balance / staining colour cast

A ``severity`` in [0, 1] scales every active knob together, giving an easy
"how bad is the capture" axis to sweep.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class PhoneConditions:
    """A reproducible phone-capture condition. All fields are 'amount of damage'."""

    brightness: float = 0.0     # 0 = unchanged; +/- fraction of exposure shift
    blur_radius: float = 0.0    # Gaussian blur radius in pixels
    jpeg_quality: int = 100     # 100 = lossless-ish; lower = more compression
    rotate_deg: float = 0.0     # rotation in degrees
    color_shift: float = 0.0    # 0 = unchanged; fraction of saturation/hue cast

    @classmethod
    def from_severity(cls, severity: float) -> "PhoneConditions":
        """Map a single 0..1 severity to a sensible bundle of degradations."""
        s = max(0.0, min(1.0, severity))
        return cls(
            brightness=0.4 * s,          # up to +/-40% exposure
            blur_radius=2.5 * s,         # up to 2.5 px blur
            jpeg_quality=int(95 - 70 * s),  # 95 -> 25
            rotate_deg=15.0 * s,         # up to 15 degrees
            color_shift=0.5 * s,         # up to 50% colour cast
        )


def apply_conditions(image: Image.Image, cond: PhoneConditions) -> Image.Image:
    """Apply a ``PhoneConditions`` bundle to a PIL image, deterministically."""
    img = image.convert("RGB")

    if cond.brightness:
        # Brightness factor < 1 darkens, > 1 brightens.
        img = ImageEnhance.Brightness(img).enhance(1.0 + cond.brightness)

    if cond.color_shift:
        # Push saturation away from neutral to mimic a white-balance/stain cast.
        img = ImageEnhance.Color(img).enhance(1.0 + cond.color_shift)

    if cond.blur_radius and cond.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=cond.blur_radius))

    if cond.rotate_deg:
        # Expand=False keeps the frame size; reflect-like fill via the edge.
        img = img.rotate(cond.rotate_deg, resample=Image.BILINEAR, expand=False)

    if cond.jpeg_quality < 100:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=max(1, int(cond.jpeg_quality)))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


def degrade(image: Image.Image, severity: float) -> Image.Image:
    """Convenience: degrade an image at a single 0..1 severity level."""
    return apply_conditions(image, PhoneConditions.from_severity(severity))
