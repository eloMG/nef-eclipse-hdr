"""Domain-specific exceptions with messages suitable for the CLI."""


class EclipseHDRError(RuntimeError):
    """Base class for expected pipeline failures."""


class DependencyError(EclipseHDRError):
    """A required runtime dependency is unavailable."""


class MetadataError(EclipseHDRError):
    """Exposure metadata is absent or inconsistent."""


class RegistrationError(EclipseHDRError):
    """Translation registration could not produce a defensible result."""


class SuspiciousAlignmentError(RegistrationError):
    """Registration completed but failed a physical sanity check."""


class OutputExistsError(EclipseHDRError):
    """A no-overwrite run found that its destination already exists."""
