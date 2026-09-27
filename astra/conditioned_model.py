"""Isolated model-server entry point; shared ImageWAM sources stay unchanged."""

from policy_server.imagewam.xpolicylab_policy.model import Model as ImageWAMModel

from astra.conditioning_audit import ConditioningAuditMixin


class Model(ConditioningAuditMixin, ImageWAMModel):
    pass
