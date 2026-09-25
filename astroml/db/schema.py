"""Backward-compatible re-export of ORM models (issue #571).

This module provides backward compatibility by re-exporting all ORM models
from astroml.db.models. New code should import directly from db.models.

Dependencies:
- astroml.db.models: ORM model definitions
"""

from astroml.db.models import *  # noqa: F401, F403
