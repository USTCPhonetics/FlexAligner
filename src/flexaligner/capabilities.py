"""Side-effect-free capability discovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import ConfigurationError, FeatureNotAvailableError


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    PLACEHOLDER = "placeholder"
    UNAVAILABLE = "unavailable"


class CapabilityId(str, Enum):
    PYTHON_API = "api.python"
    CLI = "cli"
    DISCOVERY = "capabilities.discovery"
    SINGLE_FILE_EN_CPU = "alignment.single_file.en.cpu"
    MANDARIN = "language.zh"
    GPU = "device.gpu"
    BATCH = "alignment.batch"
    WEB = "integration.web"
    AUTO_MODEL_DOWNLOAD = "models.auto_download"
    PYTHON_AUTO_MODEL_RESOLUTION = "models.auto_resolution.python"
    MULTI_FORMAT_AUDIO = "audio.multi_format"
    AUTO_RESAMPLE = "audio.auto_resample"
    CHINESE_SEGMENTATION = "text.zh_segmentation"
    DEFAULT_G2P = "pronunciation.g2p.default"
    LOCAL_ENGLISH_G2P = "pronunciation.g2p.en.local"
    CONFIDENCE_CALIBRATION = "confidence.calibration"


@dataclass(frozen=True, slots=True, kw_only=True)
class Capability:
    id: CapabilityId
    status: CapabilityStatus
    summary: str
    reason: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "id": self.id.value,
            "status": self.status.value,
            "summary": self.summary,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityReport:
    capabilities: tuple[Capability, ...]
    schema_version: int = 1

    def get(self, capability_id: CapabilityId | str) -> Capability:
        requested = (
            capability_id.value if isinstance(capability_id, CapabilityId) else capability_id
        )
        for capability in self.capabilities:
            if capability.id.value == requested:
                return capability
        raise ConfigurationError(
            f"Unknown capability {requested!r}",
            context={"capability": requested},
        )

    def require(self, capability_id: CapabilityId | str) -> Capability:
        capability = self.get(capability_id)
        if capability.status is not CapabilityStatus.AVAILABLE:
            raise FeatureNotAvailableError(
                capability.id.value,
                status=capability.status.value,
                reason=capability.reason,
            )
        return capability

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "capabilities": [item.to_dict() for item in self.capabilities],
        }


_CAPABILITIES = (
    Capability(
        id=CapabilityId.PYTHON_API,
        status=CapabilityStatus.AVAILABLE,
        summary="Import-safe Python API for strict local English CPU alignment.",
    ),
    Capability(
        id=CapabilityId.CLI,
        status=CapabilityStatus.AVAILABLE,
        summary="Help, version and capability discovery CLI.",
    ),
    Capability(
        id=CapabilityId.DISCOVERY,
        status=CapabilityStatus.AVAILABLE,
        summary="Versioned capability discovery.",
    ),
    Capability(
        id=CapabilityId.SINGLE_FILE_EN_CPU,
        status=CapabilityStatus.AVAILABLE,
        summary="English CPU single-file alignment with strict local inputs and models.",
    ),
    Capability(
        id=CapabilityId.MANDARIN,
        status=CapabilityStatus.PLACEHOLDER,
        summary="Mandarin language profile and tokenizer.",
        reason="Mandarin is outside the first implementation milestone.",
    ),
    Capability(
        id=CapabilityId.GPU,
        status=CapabilityStatus.PLACEHOLDER,
        summary="CUDA or MPS inference backend.",
        reason="Only a future GPU boundary is declared.",
    ),
    Capability(
        id=CapabilityId.BATCH,
        status=CapabilityStatus.PLACEHOLDER,
        summary="Batch alignment orchestration.",
        reason="Batch execution is outside the first implementation milestone.",
    ),
    Capability(
        id=CapabilityId.WEB,
        status=CapabilityStatus.PLACEHOLDER,
        summary="Web service adapter.",
        reason="No Web framework or server is included.",
    ),
    Capability(
        id=CapabilityId.AUTO_MODEL_DOWNLOAD,
        status=CapabilityStatus.AVAILABLE,
        summary="CLI cache resolution and confirmed download of the pinned English bundle.",
    ),
    Capability(
        id=CapabilityId.PYTHON_AUTO_MODEL_RESOLUTION,
        status=CapabilityStatus.PLACEHOLDER,
        summary="Python API automatic model resolution.",
        reason="The Python API still requires an explicit LocalModelBundle.",
    ),
    Capability(
        id=CapabilityId.MULTI_FORMAT_AUDIO,
        status=CapabilityStatus.PLACEHOLDER,
        summary="Audio formats beyond strict PCM16 WAV.",
        reason="Only the strict WAV contract is planned for the first core.",
    ),
    Capability(
        id=CapabilityId.AUTO_RESAMPLE,
        status=CapabilityStatus.PLACEHOLDER,
        summary="Automatic audio resampling.",
        reason="Implicit audio conversion is not implemented.",
    ),
    Capability(
        id=CapabilityId.CHINESE_SEGMENTATION,
        status=CapabilityStatus.PLACEHOLDER,
        summary="Chinese transcript segmentation.",
        reason="Only English whitespace tokenization is in the first core scope.",
    ),
    Capability(
        id=CapabilityId.DEFAULT_G2P,
        status=CapabilityStatus.AVAILABLE,
        summary="CLI-default local English OOV G2P with explicit warnings.",
    ),
    Capability(
        id=CapabilityId.LOCAL_ENGLISH_G2P,
        status=CapabilityStatus.AVAILABLE,
        summary="Fully local English OOV G2P backed by a pinned neural checkpoint.",
    ),
    Capability(
        id=CapabilityId.CONFIDENCE_CALIBRATION,
        status=CapabilityStatus.PLACEHOLDER,
        summary="Calibrated alignment confidence.",
        reason="Only explicitly uncalibrated raw scores are planned initially.",
    ),
)


def get_capabilities() -> CapabilityReport:
    """Return declared package capabilities without probing files or hardware."""

    return CapabilityReport(capabilities=_CAPABILITIES)
