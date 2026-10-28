
import importlib
import traceback

MAIN_PAPER = [
    ("Figure 1", "figures.fig01_architecture",           "f0_architecture"),
    ("Figure 2", "figures.fig02_two_copy_separation",    "f_fair"),
    ("Figure 3", "figures.fig03_discrete_log",           "f_dlp_wide"),
    ("Figure 4", "figures.fig04_frame_readout",          "f_modef"),
    ("Figure 5", "figures.fig05_hardware_mode_f",        "f_modef_hw"),
    ("Figure 6", "figures.fig06_trainability_law",       "f8_trainability_law"),
    ("Figure 7", "figures.fig07_representation_dial",    "f11_two_knob"),
]

SUPPLEMENTARY = [
    ("Supp. Figure 1", "figures.figS1_surplus_dimension",      "f9_surplus_dimension"),
    ("Supp. Figure 2", "figures.figS2_measurement_complexity", "f2b_exponential_separation"),
    ("Supp. Figure 3", "figures.figS3_noise_robustness",       "f_noise"),
    ("Supp. Figure 4", "figures.figS4_generalization",         "f10_generalization"),
]

ALL = MAIN_PAPER + SUPPLEMENTARY


def main() -> int:
    """Render every figure. Returns the number that failed."""
    failed = 0
    for label, module, stem in ALL:
        try:
            importlib.import_module(module).main()
            print(f"  {label:<16} {stem}.pdf")
        except Exception:
            failed += 1
            print(f"  {label:<16} {stem}.pdf  FAILED")
            traceback.print_exc()
    print(f"\n  {len(ALL) - failed} of {len(ALL)} figures rendered into figures/output/")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
