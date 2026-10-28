
set -euo pipefail

MODE="${1:-quick}"
PY="${PYTHON:-python}"

echo "=============================================================="
echo " CQFA — reproduce"
echo "=============================================================="

echo
echo "[1/4] smoke test"
$PY scripts/smoke_test.py

echo
echo "[2/4] figures from recorded results"
$PY -m figures.make_all_figures

echo
echo "[3/4] tables from recorded results"
$PY -m analysis.build_tables

echo
echo "[4/4] test suite"
$PY -m pytest tests/ experiments/spt_measurement_cost/tests/ -q

if [ "$MODE" = "--full" ]; then
  echo
  echo "=============================================================="
  echo " Re-running every simulated experiment. This overwrites"
  echo " results/ and takes hours. Compare afterwards with:"
  echo "   python scripts/verify_manifest.py"
  echo "=============================================================="
  for launcher in scripts/campaigns/0*.sh scripts/campaigns/10_*.sh; do
    echo
    echo ">>> $launcher"
    bash "$launcher"
  done
fi

if [ "$MODE" = "--with-hardware" ]; then
  echo
  echo ">>> scripts/campaigns/11_hardware_analysis.sh"
  bash scripts/campaigns/11_hardware_analysis.sh
fi

echo
echo "Done. Rendered figures are in figures/output/;"
echo "the versions used in the submitted documents are in figures/pdf/."
