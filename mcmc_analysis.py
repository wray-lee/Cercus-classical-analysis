"""
Cercus Framework — MCMC Analysis CLI
=====================================
Bayesian psychophysics analysis for multisensory integration.

Usage:
    python mcmc_analysis.py --input-dir path/to/data/ --output-dir results/
    python mcmc_analysis.py --input-dir data/ --n-chains 8 --n-draws 4000 --no-plot
"""

from __future__ import annotations

import argparse
import logging
import sys

from pipeline.mcmc import run_mcmc_analysis


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cercus MCMC — Bayesian psychophysics analysis for multisensory integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Directory containing *_session_*_events.csv and *_session_*_kinematics.csv files.",
    )
    p.add_argument(
        "--output", default="results",
        help="Directory for output files (default: results/).",
    )
    p.add_argument(
        "--binary-mode", default="escape_only",
        choices=["escape_only", "escape_prewalk"],
        help="How to binarize responses: escape_only or escape_prewalk (default: escape_only).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    # ── Sampling parameters ──
    sampling = p.add_argument_group("MCMC sampling parameters")
    sampling.add_argument(
        "--n-chains", type=int, default=4,
        help="Number of MCMC chains (default: 4).",
    )
    sampling.add_argument(
        "--n-draws", type=int, default=2000,
        help="Number of posterior draws per chain (default: 2000).",
    )
    sampling.add_argument(
        "--n-tune", type=int, default=3000,
        help="Number of tuning/warmup steps (default: 3000).",
    )

    # ── Output control ──
    output = p.add_argument_group("output control")
    output.add_argument(
        "--no-plot", action="store_true",
        help="Disable plot generation (useful for headless environments).",
    )
    output.add_argument(
        "--plot-format", default="svg", choices=["svg", "png"],
        help="Image format for plots (default: svg).",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = build_parser().parse_args(argv)

    try:
        summary = run_mcmc_analysis(
            input_dir=args.input_dir,
            output_dir=args.output,
            binary_mode=args.binary_mode,
            seed=args.seed,
            n_chains=args.n_chains,
            n_draws=args.n_draws,
            n_tune=args.n_tune,
            plot=not args.no_plot,
            plot_format=args.plot_format,
        )

        conv = summary["convergence"]
        print(f"\nAnalysis complete. Results saved to: {args.output}")
        print(
            f"Convergence: R-hat max = {conv['rhat_max']:.4f} "
            f"({'OK' if conv['rhat_ok'] else 'FAIL'}), "
            f"ESS bulk min = {conv['ess_bulk_min']:.0f} "
            f"({'OK' if conv['ess_ok'] else 'FAIL'})"
        )

        # Print key hypothesis test results
        tests = summary.get("hypothesis_tests", {})
        if tests:
            print("\nHypothesis tests (visual_only vs bimodal):")
            for key, vals in tests.items():
                if "ttc50_mean" in vals:
                    d = vals.get("cohen_d_common_scale", None)
                    prob_pos = vals.get("prob_ttc50_positive", None)
                    prob_rope = vals.get("prob_ttc50_in_rope", None)
                    hdi = vals.get("ttc50_hdi_95", [0, 0])
                    d_str = f", d={d:.2f}" if d is not None else ""
                    rope_str = ""
                    if prob_pos is not None:
                        rope_str += f", P(TTC50>0)={prob_pos:.3f}"
                    if prob_rope is not None:
                        rope_str += f", P(ROPE)={prob_rope:.3f}"
                    print(f"  {key}: mean={vals['ttc50_mean']:.1f} ms, "
                          f"HDI=[{hdi[0]:.1f}, {hdi[1]:.1f}]{d_str}{rope_str}")

    except Exception as e:
        logging.error("Analysis failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
