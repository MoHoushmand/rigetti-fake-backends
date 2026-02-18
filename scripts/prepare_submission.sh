#!/usr/bin/env bash
# =============================================================================
# prepare_submission.sh
# Prepare arXiv / IEEE TQE submission package for QRC depth optimization paper
#
# Usage:  cd qrc-depth-optimization && bash scripts/prepare_submission.sh
# Output: submission/          (clean directory)
#         submission.tar.gz    (arXiv upload archive)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAPER_DIR="paper"
SUBMIT_DIR="submission"
ARCHIVE="submission.tar.gz"
ARXIV_SIZE_LIMIT=$((50 * 1024 * 1024))   # 50 MB
ARXIV_ABSTRACT_LIMIT=1920                  # characters

PAPER_TITLE="FakeNovera and FakeCepheus: Open-Source Noise Models for Rigetti Quantum Processors with Application to Depth-Optimized Reservoir Computing"
PAPER_AUTHORS="Daniel Mo Houshmand"
PAPER_EMAIL="mo@qdaria.com"
PAPER_AFFILIATION="QDaria, Oslo, Norway"

# ---------------------------------------------------------------------------
# Colours (macOS Terminal safe)
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$*"; }

# ---------------------------------------------------------------------------
# 0. Locate project root (expect paper/ subdirectory)
# ---------------------------------------------------------------------------
if [[ ! -d "${PAPER_DIR}" ]]; then
    echo "Error: Run this script from the qrc-depth-optimization root directory."
    echo "       Expected to find ${PAPER_DIR}/ here."
    exit 1
fi

echo ""
printf "${BOLD}========================================================${NC}\n"
printf "${BOLD}  QRC Depth Optimization -- Submission Preparation${NC}\n"
printf "${BOLD}========================================================${NC}\n"
echo ""

# ---------------------------------------------------------------------------
# 1. Create / clean submission directory
# ---------------------------------------------------------------------------
info "Creating submission directory: ${SUBMIT_DIR}/"
rm -rf "${SUBMIT_DIR}" "${ARCHIVE}"
mkdir -p "${SUBMIT_DIR}/figures"
ok "Clean submission directory created."

# ---------------------------------------------------------------------------
# 2. Copy required files
# ---------------------------------------------------------------------------
info "Copying source files..."

# main.tex
cp "${PAPER_DIR}/main.tex" "${SUBMIT_DIR}/main.tex"
ok "main.tex copied."

# references.bib
cp "${PAPER_DIR}/references.bib" "${SUBMIT_DIR}/references.bib"
ok "references.bib copied."

# .bbl file (pre-compiled bibliography -- arXiv prefers this)
if [[ -f "${PAPER_DIR}/main.bbl" ]]; then
    cp "${PAPER_DIR}/main.bbl" "${SUBMIT_DIR}/main.bbl"
    ok "main.bbl copied (pre-compiled bibliography)."
else
    warn "main.bbl not found -- arXiv may need it if BibTeX is not run server-side."
fi

# Figures
FIGURE_COUNT=0
for fig in "${PAPER_DIR}"/figures/*.pdf; do
    [[ -e "$fig" ]] || continue
    cp "$fig" "${SUBMIT_DIR}/figures/"
    FIGURE_COUNT=$((FIGURE_COUNT + 1))
done
ok "${FIGURE_COUNT} PDF figures copied."

# IEEEtran.cls -- note for user
if [[ -f "${PAPER_DIR}/IEEEtran.cls" ]]; then
    cp "${PAPER_DIR}/IEEEtran.cls" "${SUBMIT_DIR}/IEEEtran.cls"
    ok "IEEEtran.cls copied from paper directory."
else
    info "IEEEtran.cls not found locally."
    info "  -> arXiv provides IEEEtran.cls automatically (v1.8b+)."
    info "  -> No action needed unless you use a custom version."
fi

# ---------------------------------------------------------------------------
# 3. Strip comments from main.tex
#    - Removes lines that are pure comments (leading %)
#    - Preserves TeX directives: \%, %% section headers, \begin{comment}, etc.
#    - Preserves inline comments after real content (conservative approach)
# ---------------------------------------------------------------------------
info "Stripping pure-comment lines from main.tex..."

ORIGINAL_LINES=$(wc -l < "${SUBMIT_DIR}/main.tex" | tr -d ' ')

# Use a temp file for safe in-place editing on macOS
TMPFILE=$(mktemp)

awk '
    # Keep blank lines
    /^[[:space:]]*$/ { print; next }
    # Keep lines starting with \  (TeX commands on comment-looking lines)
    /^[[:space:]]*\\/ { print; next }
    # Remove pure comment lines (only whitespace then %)
    /^[[:space:]]*%/ { next }
    # Keep everything else
    { print }
' "${SUBMIT_DIR}/main.tex" > "${TMPFILE}"

mv "${TMPFILE}" "${SUBMIT_DIR}/main.tex"

STRIPPED_LINES=$(wc -l < "${SUBMIT_DIR}/main.tex" | tr -d ' ')
REMOVED=$((ORIGINAL_LINES - STRIPPED_LINES))
ok "Stripped ${REMOVED} comment lines (${ORIGINAL_LINES} -> ${STRIPPED_LINES} lines)."

# ---------------------------------------------------------------------------
# 4. Flatten \input commands (if any)
# ---------------------------------------------------------------------------
info "Checking for \\input commands..."
INPUT_COUNT=$(grep -c '\\input{' "${SUBMIT_DIR}/main.tex" 2>/dev/null || true)

if [[ "${INPUT_COUNT}" -gt 0 ]]; then
    info "Found ${INPUT_COUNT} \\input commands. Flattening..."
    TMPFILE=$(mktemp)
    while IFS= read -r line; do
        if echo "$line" | grep -q '\\input{'; then
            # Extract filename from \input{filename}
            INFILE=$(echo "$line" | sed 's/.*\\input{\([^}]*\)}.*/\1/')
            # Add .tex extension if not present
            [[ "${INFILE}" == *.tex ]] || INFILE="${INFILE}.tex"
            # Try to find the file relative to paper directory
            if [[ -f "${PAPER_DIR}/${INFILE}" ]]; then
                cat "${PAPER_DIR}/${INFILE}"
                info "  Inlined: ${INFILE}"
            else
                echo "$line"
                warn "  Could not find ${INFILE} -- kept \\input command."
            fi
        else
            echo "$line"
        fi
    done < "${SUBMIT_DIR}/main.tex" > "${TMPFILE}"
    mv "${TMPFILE}" "${SUBMIT_DIR}/main.tex"
    ok "\\input flattening complete."
else
    ok "No \\input commands found -- nothing to flatten."
fi

# ---------------------------------------------------------------------------
# 5. Create tar.gz archive for arXiv
# ---------------------------------------------------------------------------
info "Creating ${ARCHIVE}..."
tar -czf "${ARCHIVE}" -C "${SUBMIT_DIR}" .
ARCHIVE_SIZE=$(stat -f%z "${ARCHIVE}" 2>/dev/null || stat --printf="%s" "${ARCHIVE}" 2>/dev/null)
ARCHIVE_SIZE_MB=$(echo "scale=2; ${ARCHIVE_SIZE} / 1048576" | bc)
ok "Archive created: ${ARCHIVE} (${ARCHIVE_SIZE_MB} MB)"

# ---------------------------------------------------------------------------
# 6. Verification checks
# ---------------------------------------------------------------------------
echo ""
printf "${BOLD}--- Verification ---${NC}\n"

# Count files
TEX_COUNT=$(find "${SUBMIT_DIR}" -name '*.tex' | wc -l | tr -d ' ')
BIB_COUNT=$(find "${SUBMIT_DIR}" -name '*.bib' | wc -l | tr -d ' ')
BBL_COUNT=$(find "${SUBMIT_DIR}" -name '*.bbl' | wc -l | tr -d ' ')
FIG_COUNT=$(find "${SUBMIT_DIR}/figures" -name '*.pdf' | wc -l | tr -d ' ')

printf "  TeX files:      %s\n" "${TEX_COUNT}"
printf "  BibTeX files:   %s\n" "${BIB_COUNT}"
printf "  BBL files:      %s\n" "${BBL_COUNT}"
printf "  PDF figures:    %s\n" "${FIG_COUNT}"

# Size check
if [[ "${ARCHIVE_SIZE}" -lt "${ARXIV_SIZE_LIMIT}" ]]; then
    ok "Archive size (${ARCHIVE_SIZE_MB} MB) is under arXiv 50 MB limit."
else
    fail "Archive size (${ARCHIVE_SIZE_MB} MB) EXCEEDS arXiv 50 MB limit!"
fi

# Abstract length check
ABSTRACT=$(awk '/\\begin\{abstract\}/,/\\end\{abstract\}/' "${SUBMIT_DIR}/main.tex" \
    | grep -v '\\begin{abstract}' | grep -v '\\end{abstract}' \
    | tr '\n' ' ' | sed 's/  */ /g')
ABSTRACT_LEN=${#ABSTRACT}
if [[ "${ABSTRACT_LEN}" -lt "${ARXIV_ABSTRACT_LIMIT}" ]]; then
    ok "Abstract length (${ABSTRACT_LEN} chars) is under arXiv ${ARXIV_ABSTRACT_LIMIT}-char limit."
else
    warn "Abstract length (${ABSTRACT_LEN} chars) may exceed arXiv ${ARXIV_ABSTRACT_LIMIT}-char limit."
    warn "Note: LaTeX markup inflates the count. Rendered text is likely shorter."
fi

# List archive contents
echo ""
printf "${BOLD}--- Archive Contents ---${NC}\n"
tar -tzf "${ARCHIVE}" | sort
echo ""

# ---------------------------------------------------------------------------
# 7. Submission checklist
# ---------------------------------------------------------------------------
printf "${BOLD}========================================================${NC}\n"
printf "${BOLD}  SUBMISSION CHECKLIST${NC}\n"
printf "${BOLD}========================================================${NC}\n"
echo ""
printf "${BOLD}arXiv Submission (https://arxiv.org/submit):${NC}\n"
echo "  [ ] Upload submission.tar.gz"
echo "  [ ] Primary category: quant-ph"
echo "  [ ] Cross-list: cs.LG, cs.ET"
echo "  [ ] All ${FIG_COUNT} figures included and referenced"
echo "  [ ] Bibliography compiles (main.bbl included)"
echo "  [ ] No external file dependencies"
echo "  [ ] Abstract under 1920 characters"
echo "  [ ] Title: ${PAPER_TITLE}"
echo "  [ ] Authors: ${PAPER_AUTHORS}"
echo "  [ ] MSC 2020 codes: 81P68 (Quantum computation), 68Q12 (Quantum computing)"
echo "  [ ] PACS codes: 03.67.Lx (Quantum computation)"
echo "  [ ] License: CC BY 4.0 or arXiv perpetual non-exclusive"
echo "  [ ] Comments field: e.g., '15 pages, 12 figures, submitted to IEEE TQE'"
echo ""

printf "${BOLD}IEEE TQE Submission (ScholarOne):${NC}\n"
echo "  URL: https://mc.manuscriptcentral.com/tqe-ieee"
echo ""
echo "  [ ] Create account / log in at ScholarOne"
echo "  [ ] Format: IEEE TQE double-column journal (IEEEtran.cls, journal option)"
echo "  [ ] Upload: PDF of compiled paper"
echo "  [ ] Upload: Source files (.tex, .bib, figures)"
echo "  [ ] Manuscript type: Regular Paper"
echo "  [ ] Cover letter (see below)"
echo "  [ ] Suggested reviewers (3-5 names)"
echo "  [ ] Keywords match IEEE taxonomy"
echo "  [ ] Conflicts of interest declared"
echo "  [ ] Data availability statement included"
echo "  [ ] Code availability: link to GitHub repository"
echo ""

# ---------------------------------------------------------------------------
# 8. Suggested cover letter
# ---------------------------------------------------------------------------
printf "${BOLD}========================================================${NC}\n"
printf "${BOLD}  SUGGESTED COVER LETTER (IEEE TQE)${NC}\n"
printf "${BOLD}========================================================${NC}\n"
cat << 'COVERLETTER'

Dear Editor,

We are pleased to submit our manuscript entitled "FakeNovera and FakeCepheus:
Open-Source Noise Models for Rigetti Quantum Processors with Application to
Depth-Optimized Reservoir Computing" for consideration as a Regular Paper in
IEEE Transactions on Quantum Engineering (TQE).

This paper makes the following contributions:

  1. We introduce the first open-source noise simulators for Rigetti's Novera
     (9-qubit) and Cepheus-1 (36-qubit) quantum processors, filling a gap in
     the fake-backend ecosystem that IBM and IQM already provide.

  2. We present a heterogeneous multi-chip noise model for Cepheus-1 that
     distinguishes intra-chip from inter-chip gate fidelities -- a feature
     absent from existing frameworks.

  3. We derive a closed-form optimal depth formula for quantum reservoir
     computing, validated through systematic simulation on Lorenz-63 chaotic
     time-series prediction.

  4. We identify a "curse of connectivity" showing that larger, more connected
     processors require shallower circuits -- a result with immediate
     implications for NISQ algorithm design.

The work combines quantum computing tooling, noise modeling, and machine
learning applications, making it well suited for TQE's interdisciplinary
readership. All code and data will be released as open-source software.

This manuscript has not been published elsewhere and is not under consideration
by another journal. All authors have approved the manuscript and agree with
its submission to IEEE TQE.

Thank you for your consideration.

Sincerely,
Daniel Mo Houshmand
QDaria, Oslo, Norway
mo@qdaria.com

COVERLETTER

echo ""
printf "${GREEN}${BOLD}Submission package ready!${NC}\n"
echo ""
echo "  Archive:   ${ARCHIVE}"
echo "  Directory: ${SUBMIT_DIR}/"
echo ""
printf "${CYAN}Next steps:${NC}\n"
echo "  1. Compile submission/main.tex locally to verify it builds:"
echo "     cd ${SUBMIT_DIR} && pdflatex main && bibtex main && pdflatex main && pdflatex main"
echo "  2. Upload ${ARCHIVE} to https://arxiv.org/submit"
echo "  3. Submit PDF + sources to https://mc.manuscriptcentral.com/tqe-ieee"
echo ""
