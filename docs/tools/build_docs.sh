#!/usr/bin/env bash
# Build the technical abstract and the proof-of-concept PDF from one run.
#   docs/tools/build_docs.sh <run> [--team-id ID] [--viewer-url URL] [--repo-url URL] [--video-url URL]
# The final submission uses the run named "room":  docs/tools/build_docs.sh room --team-id ... 
# Without --team-id and --viewer-url the PDFs carry clearly marked slots, never made-up links.
# Output: docs/build/01_Technical_Abstract[_dryrun].pdf and docs/build/00_START_HERE[_dryrun].pdf
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${1:?usage: docs/tools/build_docs.sh <run> [--team-id ID] [--viewer-url URL] [--repo-url URL] [--video-url URL]}
shift
T="$ROOT/docs/tools"; B="$ROOT/docs/build"
[ -f "$ROOT/runs/$RUN/report.json" ] || { echo "no run record: runs/$RUN/report.json" >&2; exit 1; }
python3 "$T/doc_values.py" "$RUN" "$@"
python3 "$T/make_figures.py" "$RUN"
# the README results block follows the run named on the command line, unless told not to
[ -n "${NS_NO_README:-}" ] || python3 "$T/update_readme.py" "$RUN"
SUFFIX=""
[ "$RUN" = "room" ] || SUFFIX="_dryrun"
{ [ "$RUN" = "room" ] || [ "$RUN" = "rh-honka" ]; } || SUFFIX="_dryrun_$RUN"
V="$B/$RUN/values.json"
python3 "$T/render_doc.py" "$ROOT/docs/abstract/template.html" "$B/01_Technical_Abstract$SUFFIX.pdf" --data "$V" --max-pages 5
python3 "$T/render_doc.py" "$ROOT/docs/poc/00_START_HERE_Proof_of_Concept_and_Links.html" "$B/00_START_HERE$SUFFIX.pdf" --data "$V" --max-pages 2
for f in "$B/01_Technical_Abstract$SUFFIX.pdf" "$B/00_START_HERE$SUFFIX.pdf"; do
  echo "$(basename "$f"): $(pdfinfo "$f" | grep Pages)"
done
