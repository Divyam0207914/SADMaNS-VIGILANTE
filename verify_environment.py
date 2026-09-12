"""
SADMaNS — Environment Verification Script
==========================================

Run:  python verify_environment.py

Checks that all required dependencies are installed and functional.
Also performs a minimal PyTorch Geometric GCNConv smoke test
and a mitreattack-python import test.
"""

import importlib
import os
import sys


def _check_import(module_name: str, display_name: str | None = None) -> tuple[bool, str]:
    """Try to import a module. Return (success, version_or_error)."""
    display = display_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except Exception as e:
        return False, str(e)


def _sep(char: str = "=", width: int = 56) -> str:
    return char * width


def main() -> int:
    print(_sep())
    print("  SADMaNS ENVIRONMENT CHECK")
    print(_sep())
    print()

    all_ok = True
    results: list[tuple[str, bool, str]] = []

    # ── Python ──
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    results.append(("Python", py_ok, py_ver))
    if not py_ok:
        all_ok = False

    # ── Core packages ──
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("sklearn", "Scikit-learn"),
        ("networkx", "NetworkX"),
        ("plotly", "Plotly"),
        ("matplotlib", "Matplotlib"),
        ("streamlit", "Streamlit"),
    ]

    for mod_name, display in packages:
        ok, info = _check_import(mod_name, display)
        results.append((display, ok, info))
        if not ok:
            all_ok = False

    # ── MITRE ATT&CK ──
    mitre_ok = False
    mitre_info = ""
    try:
        from mitreattack.stix20 import MitreAttackData  # noqa: F401
        import mitreattack
        mitre_info = getattr(mitreattack, "__version__", "imported OK")
        mitre_ok = True
    except ImportError:
        try:
            # Fallback: maybe the package restructured
            import mitreattack  # noqa: F401
            mitre_info = getattr(mitreattack, "__version__", "imported OK (top-level)")
            mitre_ok = True
        except Exception as e:
            mitre_info = str(e)
    results.append(("MITRE ATT&CK", mitre_ok, mitre_info))
    if not mitre_ok:
        all_ok = False

    # ── Print results table ──
    max_name = max(len(r[0]) for r in results)
    for name, ok, info in results:
        status = "OK" if ok else "FAILED"
        pad = " " * (max_name - len(name) + 2)
        version_str = f"  ({info})" if ok and info != "unknown" else ""
        fail_str = f"\n{'':>{max_name + 4}}REASON: {info}" if not ok else ""
        print(f"  {name}:{pad}{status}{version_str}{fail_str}")

    # ── Device info ──
    print()
    print("  Device:")
    try:
        import torch
        cuda = torch.cuda.is_available()
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"    CUDA available:    {cuda}")
        print(f"    MPS available:     {mps}")
        if cuda:
            print(f"    Selected device:   cuda")
        elif mps:
            print(f"    Selected device:   mps")
        else:
            print(f"    Selected device:   cpu")
    except Exception as e:
        print(f"    Could not detect device: {e}")

    # ── Dataset check ──
    print()
    print("  Dataset:")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "02-15-2018.csv")
    if os.path.isfile(data_path):
        size_mb = os.path.getsize(data_path) / (1024 ** 2)
        print(f"    02-15-2018.csv:    FOUND  ({size_mb:.0f} MB)")
    else:
        print(f"    02-15-2018.csv:    NOT FOUND")
        print(f"    Expected at:       {data_path}")
        all_ok = False

    # ── PyTorch Geometric smoke test ──
    print()
    print("  PyG GCNConv Smoke Test:")
    try:
        import torch
        from torch_geometric.nn import GCNConv, global_mean_pool
        from torch_geometric.data import Data

        # Tiny graph: 3 nodes, 3 edges (triangle), 4 features per node.
        x = torch.randn(3, 4)
        edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                                   [1, 0, 2, 1, 2, 0]], dtype=torch.long)
        batch = torch.zeros(3, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        conv = GCNConv(4, 8)
        out = conv(data.x, data.edge_index)
        pooled = global_mean_pool(out, batch)

        assert out.shape == (3, 8), f"Expected (3, 8), got {out.shape}"
        assert pooled.shape == (1, 8), f"Expected (1, 8), got {pooled.shape}"
        print(f"    GCNConv(4 -> 8):   OK  (output shape: {tuple(out.shape)})")
        print(f"    Global pool:       OK  (output shape: {tuple(pooled.shape)})")
    except Exception as e:
        print(f"    FAILED: {e}")
        all_ok = False

    # ── MITRE ATT&CK API test ──
    print()
    print("  MITRE ATT&CK API Test:")
    try:
        from mitreattack.stix20 import MitreAttackData
        print(f"    MitreAttackData:   IMPORTABLE")
        print(f"    (Full STIX data download deferred to implementation phase)")
    except ImportError as e:
        # Check if the module exists under a different path
        try:
            import mitreattack
            available = dir(mitreattack)
            print(f"    mitreattack imported, available attrs: {available[:10]}...")
        except Exception:
            print(f"    FAILED: {e}")
            all_ok = False

    # ── Config test ──
    print()
    print("  Config:")
    try:
        import config
        print(f"    DATA_PATH:         {config.DATA_PATH}")
        print(f"    WINDOW_SIZE:       {config.WINDOW_SIZE}")
        print(f"    SEQ_LEN:           {config.SEQ_LEN}")
        print(f"    GRAPH_MODE:        {config.GRAPH_MODE}")
        print(f"    TOP_K_PORTS:       {config.TOP_K_PORTS}")
        print(f"    GCN_HIDDEN_DIM:    {config.GCN_HIDDEN_DIM}")
        print(f"    GRU_HIDDEN_DIM:    {config.GRU_HIDDEN_DIM}")
        print(f"    GRU_INPUT_DIM:     {config.GRU_INPUT_DIM}")
        print(f"    NODE_FEATURE_DIM:  {config.NODE_FEATURE_DIM}")
        print(f"    GLOBAL_STATE_DIM:  {config.GLOBAL_STATE_DIM}")
        print(f"    DEVICE:            {config.DEVICE}")

        issues = config.validate_config()
        if issues:
            print(f"    Validation issues:")
            for issue in issues:
                print(f"      ⚠  {issue}")
        else:
            print(f"    Validation:        ALL OK")
    except Exception as e:
        print(f"    FAILED: {e}")
        all_ok = False

    # ── Summary ──
    print()
    print(_sep())
    if all_ok:
        print("  ✓  ALL CHECKS PASSED — Environment ready.")
    else:
        print("  ✗  SOME CHECKS FAILED — See details above.")
    print(_sep())
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
