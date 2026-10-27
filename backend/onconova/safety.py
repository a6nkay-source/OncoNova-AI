"""Safety labels and the guard that keeps device-control code out of the process.

OncoNova AI is a research prototype. It never controls, configures or talks to a
linear accelerator or any other medical device, and it never produces a radiation
prescription. This module holds the labels every response carries and a runtime
check that refuses to start if a device-communication library has been imported.
tests/test_safety.py enforces the same rule statically over the source tree.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any

RESEARCH_DISCLAIMER = (
    "Research and educational prototype. Not a medical device. Not for diagnosis, "
    "treatment planning or any clinical decision."
)
DEMO_LABEL = "DEMO DATA"
SIMULATION_LABEL = (
    "SIMULATED - NON-CLINICAL. Geometric illustration on synthetic or research data; "
    "not a dose calculation and not a treatment plan."
)
DEMO_MODEL_LABEL = "DEMO MODEL - trained or tuned on synthetic phantoms only; not validated on patients."

# Libraries for serial ports, USB, instrument buses, industrial fieldbuses and DICOM
# networking. None is a dependency; importing any of them is treated as a defect.
FORBIDDEN_MODULES = frozenset({
    "serial", "usb", "pyvisa", "pymodbus", "modbus_tk", "can", "pynetdicom", "hl7apy",
    "pyftdi", "bluetooth", "bleak",
})


class EvidenceKind(str, Enum):
    """How a statement shown to the user was produced."""

    MEASURED = "measured"                 # computed directly from images or reference masks
    MEASURED_FROM_MODEL = "measured_from_model_output"  # a measurement of an AI segmentation
    MODEL_OUTPUT = "model_output"         # an AI prediction or hypothesis
    UNCERTAIN = "uncertain"               # low-confidence or out-of-distribution output
    EXPERT_REVIEW = "requires_expert_review"
    SIMULATED = "simulated"               # synthetic, illustrative, non-clinical
    UNAVAILABLE = "unavailable"           # required inputs missing; nothing was predicted


def finding(kind: EvidenceKind, text: str, **extra: Any) -> dict[str, Any]:
    return {"kind": kind.value, "text": text, **extra}


def unavailable(reason: str, *, missing: list[str] | None = None, how_to_fix: str | None = None) -> dict[str, Any]:
    """Explicit unavailable state returned instead of an unsupported prediction."""
    return {
        "status": "unavailable",
        "kind": EvidenceKind.UNAVAILABLE.value,
        "reason": reason,
        "missing_inputs": missing or [],
        "how_to_fix": how_to_fix,
    }


class DeviceInterfaceDetected(RuntimeError):
    pass


def assert_no_device_interfaces() -> None:
    loaded = sorted(name for name in FORBIDDEN_MODULES if name in sys.modules)
    if loaded:
        raise DeviceInterfaceDetected(
            f"Refusing to start: device-communication modules are loaded ({', '.join(loaded)}). "
            "OncoNova AI must never communicate with medical equipment."
        )
