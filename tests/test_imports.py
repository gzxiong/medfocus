"""Smoke-test that every public module imports without side effects."""

from __future__ import annotations


def test_top_level():
    import medfocus
    assert hasattr(medfocus, "MedFocus")
    assert hasattr(medfocus, "load_lvlm")
    assert hasattr(medfocus, "load_medground")


def test_submodules():
    import medfocus.attribution.eval as e
    import medfocus.attribution.medfocus as m
    import medfocus.concepts.intervention as ci
    import medfocus.concepts.transfer as ct
    import medfocus.data.io
    import medfocus.lvlm.adapters as la
    import medfocus.lvlm.registry as lr
    import medfocus.medsam.client as mc
    import medfocus.ot.mapping as om
    import medfocus.ot.reference as orf
    import medfocus.ot.sinkhorn as os
    import medfocus.utils.tokens as ut

    assert all([e, m, ci, ct, la, lr, mc, om, orf, os, ut])
