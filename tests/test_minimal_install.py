"""What has to keep working on a minimal install.

`requirements-minimal.txt` deliberately omits cargen, ultralytics, plate OCR,
scikit-learn and matplotlib. That configuration is documented in the README as
supported, so the paths it changes need to degrade rather than error -- a 500
on the advertised install is worse than not offering it.
"""
import sys

import pytest
from fastapi.testclient import TestClient

from server.api import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_url=f"sqlite:///{tmp_path}/m.sqlite",
                     crops_dir=str(tmp_path / "crops"),
                     targets3d_dir=str(tmp_path / "t3d"))
    with TestClient(app) as c:
        yield c


def _flag(client, label="silver camry - case 12"):
    r = client.post("/api/targets", json={"label": label})
    assert r.status_code == 201, r.text
    return r.json()["target_id"]


def test_model3d_reports_absent_bridge_instead_of_500(client, monkeypatch):
    """car3d imports cargen at module scope, and a minimal install has no
    cargen. Unguarded, the dossier's 3D call raised through as a 500 on exactly
    the configuration we tell people to use. As far as the dossier is
    concerned "no 3D bridge" is the same answer as "nothing fused yet".
    """
    target_id = _flag(client)
    # None in sys.modules makes `import x` raise ImportError -- the same shape
    # of failure as the package genuinely not being installed.
    monkeypatch.setitem(sys.modules, "car3d.geometry", None)
    monkeypatch.setitem(sys.modules, "car3d.profile_model", None)

    r = client.get(f"/api/targets/{target_id}/model3d")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exists"] is False
    # Say WHY, so a missing 3D panel is distinguishable from an empty one.
    assert "3D bridge unavailable" in body.get("reason", "")


def test_unknown_target_still_404s_with_no_bridge(client, monkeypatch):
    """The bridge guard must not swallow the identity check above it: a typo'd
    target id is a 404, not a cheerful exists=false."""
    monkeypatch.setitem(sys.modules, "car3d.geometry", None)
    monkeypatch.setitem(sys.modules, "car3d.profile_model", None)
    assert client.get("/api/targets/tgt-nope/model3d").status_code == 404


def test_calibration_runtime_needs_no_sklearn():
    """scikit-learn is dropped from the minimal set because it is only needed
    to FIT the isotonic curve. The runtime reads a precomputed grid, so the
    cascade is calibrated identically -- if this ever stops holding, the
    minimal install silently starts scoring differently from the full one.
    """
    import calibration.isotonic as iso

    src = (iso.__file__ or "")
    assert src, "isotonic module has no source path"
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    # Any sklearn import must sit inside a function, not at module scope.
    for line in text.splitlines():
        if line.startswith(("import sklearn", "from sklearn")):
            raise AssertionError(
                "sklearn imported at module scope: the runtime is supposed to "
                "stay sklearn-free so requirements-minimal.txt can drop it")


def test_minimal_requirements_file_stays_in_sync():
    """Every pin in the minimal file must also appear in the full one, at the
    same version. Two files that drift produce an install that resolves
    differently from the one the numbers were measured on."""
    def pins(path):
        out = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line or "==" not in line:
                    continue
                name, _, ver = line.partition("==")
                out[name.strip()] = ver.strip()
        return out

    full = pins("requirements.txt")
    minimal = pins("requirements-minimal.txt")
    assert minimal, "minimal requirements file parsed as empty"
    drift = {k: (v, full.get(k)) for k, v in minimal.items()
             if k in full and full[k] != v}
    assert not drift, f"version drift between requirements files: {drift}"
    missing = [k for k in minimal if k not in full]
    assert not missing, (
        f"{missing} is in requirements-minimal.txt but not requirements.txt, "
        f"so the full install would not include it")
