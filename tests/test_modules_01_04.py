"""Pytest unit and integration tests for Modules 01-04."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from core import (
    cellular_automata as m03,
    evolutionary_ga as m04,
    info_theory_pid as m01,
    real_world_demos as demos,
    transfer_entropy as m02,
)


def test_module01_identities_and_gates() -> None:
    land = m01.landauer_erasure_energy(300.0, 1.0)
    assert abs(land["e_bit_joules"] - 2.8712e-21) < 1e-23

    nl = m01.circular_nonlinearity_benchmark(n_samples=1200, seed=42)
    assert abs(nl["pearson_r"]) < 0.08
    assert nl["mi_ksg_bits"] > 0.3

    xor_r = m01.evaluate_mechanism_pid("XOR", sigma=0.0, coupling=1.0)
    and_r = m01.evaluate_mechanism_pid("AND", sigma=0.0, coupling=1.0)
    copy_r = m01.evaluate_mechanism_pid("COPY", sigma=0.0, coupling=1.0)
    ident_r = m01.evaluate_mechanism_pid("IDENTICAL", sigma=0.0, coupling=1.0)
    assert abs(xor_r["Synergy"] - 1.0) < 1e-6
    assert abs(and_r["Synergy"] - 0.5) < 1e-3
    assert abs(and_r["Redundancy"] - 0.311278) < 1e-3
    assert abs(copy_r["Unique_X1"] - 1.0) < 1e-6
    assert abs(ident_r["Redundancy"] - 1.0) < 1e-6


def test_module02_te_delay_and_fdr() -> None:
    x, y, z = m02.generate_synthetic_motif("Direct_Coupling", n_steps=450, coupling=0.70, seed=42)
    te_u1 = m02.compute_transfer_entropy(x, y, z, lag=1, k=5)
    te_u2 = m02.compute_transfer_entropy(x, y, z, lag=2, k=5)
    assert te_u2 > te_u1 and te_u2 > 0.08

    fdr = m02.benjamini_hochberg_fdr([0.001, 0.012, 0.45, 0.80], alpha=0.05)
    assert list(fdr["rejected"]) == [True, True, False, False]


def test_module03_fst_and_damage() -> None:
    assert abs(m03.compute_langton_lambda(54) - 0.5) < 1e-9
    grid, _ = m03.evolve_eca(54, width=128, steps=80, init_type="collision_setup", seed=42)
    fst = m03.HansonCrutchfieldFST()
    mask, states = fst.filter(grid)
    cat = m03.extract_collision_catalog(mask)
    assert len(cat["collision_vertices"]) >= 1
    assert set(int(v) for v in set(states.ravel())).issubset({0, 1, 2, 3, 4})


def test_module04_gkl_and_schema() -> None:
    gkl = m04.build_gkl_rule()
    ic_hi = m04.make_density_ic(width=149, init_density=0.56, seed=101)
    st = m04.simulate_r3_ca(gkl, ic_hi, max_steps=160)
    assert float(st[-1].mean()) == 1.0

    o_h, d_h = m04.schema_properties("1**0*1**")
    assert (o_h, d_h) == (3, 5)
    bnd = m04.holland_schema_lower_bound(4, 0.78, 0.52, d_h, o_h)
    assert bnd > 4.0


def test_all_12_real_world_demos_run() -> None:
    for fn in (
        demos.demo_m01_computational_neuroscience,
        demos.demo_m01_genomics_epistasis,
        demos.demo_m01_thermodynamic_reversible_computing,
        demos.demo_m02_brain_connectivity,
        demos.demo_m02_finance_contagion,
        demos.demo_m02_gene_regulatory_network,
        demos.demo_m03_excitable_bz_spiral,
        demos.demo_m03_traffic_flow_rule184,
        demos.demo_m03_neuromorphic_glider_logic,
        demos.demo_m04_protein_directed_evolution,
        demos.demo_m04_automl_nas,
        demos.demo_m04_swarm_robotics_foraging,
    ):
        fig, _ = fn()
        plt.close(fig)


def test_streamlit_app_renders_all_modules() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    for option in [
        "Module 01: Info Theory & PID",
        "Module 02: Transfer Entropy",
        "Module 03: Cellular Automata",
        "Module 04: Evolutionary GA",
    ]:
        at.sidebar.radio[0].set_value(option).run()
        assert not at.exception


def test_notebooks_01_to_04_headless_import() -> None:
    import json
    import types

    shim = types.ModuleType("ipywidgets")
    shim.interact = lambda *a, **kw: None
    shim.Dropdown = lambda *a, **kw: None
    shim.FloatSlider = lambda *a, **kw: None
    shim.IntSlider = lambda *a, **kw: None
    shim.Checkbox = lambda *a, **kw: None
    shim.fixed = lambda *a, **kw: None
    old = sys.modules.get("ipywidgets")
    sys.modules["ipywidgets"] = shim
    try:
        for nb_name in (
            "01_information_theory_and_pid.ipynb",
            "02_transfer_entropy_and_network_inference.ipynb",
            "03_ca_and_particle_computation.ipynb",
            "04_evolutionary_computation_holland.ipynb",
        ):
            nb_path = REPO_ROOT / "notebooks" / nb_name
            nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
            ns = {"__name__": f"test_{nb_path.stem}"}
            for cell in nb_data["cells"]:
                if cell["cell_type"] == "code":
                    src = "".join(cell["source"])
                    exec(compile(src, str(nb_path), "exec"), ns)
    finally:
        if old is not None:
            sys.modules["ipywidgets"] = old
        else:
            sys.modules.pop("ipywidgets", None)

