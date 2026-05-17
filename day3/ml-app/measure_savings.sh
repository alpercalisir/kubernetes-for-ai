#!/bin/bash

# ============================================================
# compare_dockerfiles.sh
#
# Compares ml-app:day2 vs ml-app:final
# Shows exactly which new line in the optimized Dockerfile
# saved how many MB — measured from real file deletions.
#
# No rebuilds. Uses your two existing images only.
# Usage: chmod +x compare_dockerfiles.sh && ./compare_dockerfiles.sh
# ============================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'
DIM='\033[2m'

mb() { awk "BEGIN{printf \"%.2f\", $1/1024/1024}"; }

bar() {
    local val=$1 max=$2 width=${3:-35}
    local filled=$(awk "BEGIN{v=int($val/$max*$width); print (v<1 && $val>0)?1:v}")
    local empty=$(( width - filled ))
    [ $filled -gt 0 ] && printf '%0.s█' $(seq 1 $filled)
    [ $empty  -gt 0 ] && printf '%0.s░' $(seq 1 $empty)
}

# ── check images exist ─────────────────────────────────────

for img in ml-app:day2 ml-app:final; do
    if ! docker image inspect "$img" > /dev/null 2>&1; then
        echo -e "${RED}✗ $img not found.${RESET}"
        exit 1
    fi
done

clear
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  DOCKERFILE LINE SAVINGS                                                    ║${RESET}"
echo -e "${BOLD}${CYAN}║  ml-app:day2 (126MB)  →  ml-app:final (95.8MB)                             ║${RESET}"
echo -e "${BOLD}${CYAN}║  Each new line in your optimized Dockerfile, measured in real MB            ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── scan both images ───────────────────────────────────────

echo -e "${BOLD}  Step 1/3 — Scanning files inside both images${RESET}"
echo ""

echo -ne "  ml-app:day2  ... "
docker run --rm --entrypoint="" ml-app:day2 \
    find /usr/local/lib -type f -printf '%s %p\n' 2>/dev/null \
    | sort -k2 > /tmp/d2.txt
D2_BYTES=$(awk '{s+=$1}END{print s+0}' /tmp/d2.txt)
D2_FILES=$(wc -l < /tmp/d2.txt)
echo -e "${GREEN}done${RESET}   ${BOLD}$(mb $D2_BYTES) MB${RESET}   ${DIM}${D2_FILES} files${RESET}"

echo -ne "  ml-app:final ... "
docker run --rm --entrypoint="" ml-app:final \
    find /usr/local/lib -type f -printf '%s %p\n' 2>/dev/null \
    | sort -k2 > /tmp/df.txt
DF_BYTES=$(awk '{s+=$1}END{print s+0}' /tmp/df.txt)
DF_FILES=$(wc -l < /tmp/df.txt)
echo -e "${GREEN}done${RESET}   ${BOLD}$(mb $DF_BYTES) MB${RESET}   ${DIM}${DF_FILES} files${RESET}"

echo ""

# ── diff: deleted files + shrunk files ────────────────────

echo -e "${BOLD}  Step 2/3 — Finding what changed${RESET}"
echo ""

awk '{print $2}' /tmp/d2.txt | sort > /tmp/d2_paths.txt
awk '{print $2}' /tmp/df.txt | sort > /tmp/df_paths.txt

# deleted = in day2 only
comm -23 /tmp/d2_paths.txt /tmp/df_paths.txt > /tmp/deleted_paths.txt
DELETED_FILES=$(wc -l < /tmp/deleted_paths.txt)
DELETED_BYTES=$(grep -Ff /tmp/deleted_paths.txt /tmp/d2.txt 2>/dev/null | awk '{s+=$1}END{print s+0}')

# shrunk = in both but smaller in final (strip --strip-all effect)
> /tmp/shrunk.txt
comm -12 /tmp/d2_paths.txt /tmp/df_paths.txt | while read -r path; do
    s1=$(grep -F " ${path}$" /tmp/d2.txt | awk '{print $1}' | head -1)
    s2=$(grep -F " ${path}$" /tmp/df.txt | awk '{print $1}' | head -1)
    if [ -n "$s1" ] && [ -n "$s2" ] && [ "$s1" -gt "$s2" ] 2>/dev/null; then
        echo "$(( s1 - s2 )) $path"
    fi
done > /tmp/shrunk.txt
SHRUNK_FILES=$(wc -l < /tmp/shrunk.txt)
SHRUNK_BYTES=$(awk '{s+=$1}END{print s+0}' /tmp/shrunk.txt)

TOTAL_SAVED=$(( DELETED_BYTES + SHRUNK_BYTES ))
TOTAL_PCT=$(awk "BEGIN{printf \"%.1f\", ($TOTAL_SAVED/$D2_BYTES)*100}")

echo -e "  Files deleted        : ${RED}${DELETED_FILES} files${RESET}   $(mb $DELETED_BYTES) MB"
echo -e "  Files shrunk (strip) : ${YELLOW}${SHRUNK_FILES} files${RESET}   $(mb $SHRUNK_BYTES) MB"
echo -e "  Total saved          : ${GREEN}$(mb $TOTAL_SAVED) MB${RESET}   (${TOTAL_PCT}%)"
echo ""

# ── define each Dockerfile line with its file pattern ─────

echo -e "${BOLD}  Step 3/3 — Mapping to each Dockerfile line${RESET}"
echo ""

# Format: NAME | DESCRIPTION | PATTERN | TYPE(D=deleted / S=shrunk)
declare -a NAMES DESCS PATTERNS TYPES

NAMES[0]="find -type d -name 'tests'"
DESCS[0]="sklearn + numpy unit test directories"
PATTERNS[0]="/tests/"
TYPES[0]="D"

NAMES[1]="find -type d -name 'test'"
DESCS[1]="remaining test directories"
PATTERNS[1]="(^|/)test/"
TYPES[1]="D"

NAMES[2]="find -type d -name '__pycache__'"
DESCS[2]="Python bytecode cache directories"
PATTERNS[2]="/__pycache__/"
TYPES[2]="D"

NAMES[3]="find -name '*.pyc'"
DESCS[3]="compiled .pyc bytecode files"
PATTERNS[3]="\.pyc$"
TYPES[3]="D"

NAMES[4]="find -name '*.pyo'"
DESCS[4]="optimised .pyo bytecode files"
PATTERNS[4]="\.pyo$"
TYPES[4]="D"

NAMES[5]="find -name '*.pyd'"
DESCS[5]="Windows Python extension files"
PATTERNS[5]="\.pyd$"
TYPES[5]="D"

NAMES[6]="find -name '*.dist-info' (excl. Flask)"
DESCS[6]="dist-info for sklearn/numpy/pip etc"
PATTERNS[6]="\.dist-info/"
TYPES[6]="D"

NAMES[7]="find -name '*.egg-info' (excl. Flask)"
DESCS[7]="egg-info metadata directories"
PATTERNS[7]="\.egg-info/"
TYPES[7]="D"

NAMES[8]="find -type d -name 'examples'"
DESCS[8]="bundled example scripts"
PATTERNS[8]="/examples/"
TYPES[8]="D"

NAMES[9]="find -type d -name 'docs'"
DESCS[9]="docs/ documentation directories"
PATTERNS[9]="/docs/"
TYPES[9]="D"

NAMES[10]="find -type d -name 'doc'"
DESCS[10]="doc/ alternate documentation dirs"
PATTERNS[10]="(^|/)doc/"
TYPES[10]="D"

NAMES[11]="find -type d -name 'data'"
DESCS[11]="bundled data directories"
PATTERNS[11]="(^|/)data/"
TYPES[11]="D"

NAMES[12]="find -name '*.pyx'"
DESCS[12]="Cython source files"
PATTERNS[12]="\.pyx$"
TYPES[12]="D"

NAMES[13]="find -name '*.pxd'"
DESCS[13]="Cython header declaration files"
PATTERNS[13]="\.pxd$"
TYPES[13]="D"

NAMES[14]="find -name '*.c'"
DESCS[14]="C source files (already compiled to .so)"
PATTERNS[14]="\.c$"
TYPES[14]="D"

NAMES[15]="strip --strip-all *.so"
DESCS[15]="debug symbols from compiled .so libs"
PATTERNS[15]="\.so"
TYPES[15]="S"

NAMES[16]="clean *.dist-info extra files"
DESCS[16]="metadata files inside kept dist-info"
PATTERNS[16]="\.dist-info/.+"
TYPES[16]="D"

# ── measure each line ──────────────────────────────────────

cp /tmp/deleted_paths.txt /tmp/pool_del.txt
cp /tmp/shrunk.txt        /tmp/pool_shr.txt

declare -a L_BYTES L_FILES
MAX_BYTES=1

for i in "${!NAMES[@]}"; do
    PAT="${PATTERNS[$i]}"
    TYPE="${TYPES[$i]}"

    if [ "$TYPE" = "D" ]; then
        MATCHED=$(grep -E "$PAT" /tmp/pool_del.txt 2>/dev/null || true)
        if [ -n "$MATCHED" ]; then
            BYTES=$(echo "$MATCHED" | grep -Ff - /tmp/d2.txt 2>/dev/null | awk '{s+=$1}END{print s+0}')
            COUNT=$(echo "$MATCHED" | wc -l | tr -d ' ')
        else
            BYTES=0; COUNT=0
        fi
        grep -vE "$PAT" /tmp/pool_del.txt > /tmp/pool_del_tmp.txt 2>/dev/null || true
        mv /tmp/pool_del_tmp.txt /tmp/pool_del.txt
    else
        BYTES=$(grep -E "$PAT" /tmp/pool_shr.txt 2>/dev/null | awk '{s+=$1}END{print s+0}')
        COUNT=$(grep -cE "$PAT" /tmp/pool_shr.txt 2>/dev/null || echo 0)
        grep -vE "$PAT" /tmp/pool_shr.txt > /tmp/pool_shr_tmp.txt 2>/dev/null || true
        mv /tmp/pool_shr_tmp.txt /tmp/pool_shr.txt
    fi

    L_BYTES[$i]=${BYTES:-0}
    L_FILES[$i]=${COUNT:-0}
    [ "${L_BYTES[$i]}" -gt "$MAX_BYTES" ] && MAX_BYTES="${L_BYTES[$i]}"
done

# ── results table ──────────────────────────────────────────

SEP=$(printf '─%.0s' {1..118})

echo -e "${BOLD}${CYAN}${SEP}${RESET}"
echo -e "${BOLD}  RESULTS — each line of your optimized Dockerfile vs Day2${RESET}"
echo -e "${BOLD}${CYAN}${SEP}${RESET}"
echo ""
printf "  ${BOLD}%-40s  %-40s  %9s  %12s  %-35s${RESET}\n" \
    "Dockerfile line" "What it removes" "Saved MB" "File count" "Bar"
printf "  ${DIM}%s${RESET}\n" "$SEP"

# baseline
printf "  ${YELLOW}%-40s${RESET}  %-40s  ${YELLOW}%8sMB${RESET}\n" \
    "Day2: pip install (no cleanup)" "all packages installed" "$(mb $D2_BYTES)"
printf "  ${DIM}%s${RESET}\n" "$SEP"

CUMUL=0
for i in "${!NAMES[@]}"; do
    B="${L_BYTES[$i]}"
    F="${L_FILES[$i]}"
    CUMUL=$(( CUMUL + B ))

    if   [ "$B" -ge $((5*1024*1024)) ];  then COLOR=$GREEN;  STAR="★★★"
    elif [ "$B" -ge $((512*1024)) ];     then COLOR=$YELLOW; STAR="★★ "
    elif [ "$B" -ge $((1024)) ];         then COLOR=$CYAN;   STAR="★  "
    else                                      COLOR=$DIM;    STAR="   "; fi

    printf "  ${COLOR}%s${RESET} ${BOLD}%-37s${RESET}  ${DIM}%-40s${RESET}  ${COLOR}%+8sMB${RESET}  ${DIM}%10s files${RESET}  ${COLOR}%s${RESET}\n" \
        "$STAR" "${NAMES[$i]}" "${DESCS[$i]}" \
        "-$(mb $B)" "$F" "$(bar $B $MAX_BYTES 35)"
done

printf "  ${DIM}%s${RESET}\n" "$SEP"
printf "  ${BOLD}%-40s  %-40s  ${GREEN}%+8sMB${RESET}${BOLD}  %11s${RESET}\n" \
    "Final: ml-app:final" "" \
    "-$(mb $TOTAL_SAVED)" "${TOTAL_PCT}% saved"

# ── ranked table ───────────────────────────────────────────

echo ""
echo -e "${BOLD}${CYAN}${SEP}${RESET}"
echo -e "${BOLD}  RANKED — biggest wins first${RESET}"
echo -e "${BOLD}${CYAN}${SEP}${RESET}"
echo ""
printf "  ${BOLD}%-20s  %-40s  %-40s  %9s  %12s${RESET}\n" \
    "Impact" "Dockerfile line" "What it removes" "Saved MB" "File count"
printf "  ${DIM}%s${RESET}\n" "$SEP"

declare -a RANK
for i in "${!NAMES[@]}"; do
    RANK+=("${L_BYTES[$i]}|${NAMES[$i]}|${DESCS[$i]}|${L_FILES[$i]}")
done
IFS=$'\n' SORTED=($(printf '%s\n' "${RANK[@]}" | sort -t'|' -k1 -rn))
unset IFS

for entry in "${SORTED[@]}"; do
    B=$(echo  "$entry" | cut -d'|' -f1)
    NM=$(echo "$entry" | cut -d'|' -f2)
    DS=$(echo "$entry" | cut -d'|' -f3)
    FC=$(echo "$entry" | cut -d'|' -f4)

    if   [ "$B" -ge $((5*1024*1024)) ];  then COLOR=$GREEN;  BADGE="★★★  HIGH IMPACT  "
    elif [ "$B" -ge $((512*1024)) ];     then COLOR=$YELLOW; BADGE="★★   MEDIUM      "
    elif [ "$B" -ge $((1024)) ];         then COLOR=$CYAN;   BADGE="★    LOW         "
    else                                      COLOR=$DIM;    BADGE="     NEGLIGIBLE  "; fi

    printf "  ${COLOR}%-20s${RESET}  ${BOLD}%-40s${RESET}  ${DIM}%-40s${RESET}  ${COLOR}-%8sMB${RESET}  ${DIM}%10s files${RESET}\n" \
        "$BADGE" "$NM" "$DS" "$(mb $B)" "$FC"
done

# ── what remains ───────────────────────────────────────────

echo ""
echo -e "${BOLD}${CYAN}${SEP}${RESET}"
echo -e "${BOLD}  WHAT REMAINS IN ml-app:final  (cannot be removed safely)${RESET}"
echo -e "${BOLD}${CYAN}${SEP}${RESET}"
echo ""

docker run --rm --entrypoint="" ml-app:final \
    du -sh /usr/local/lib/python3.11/site-packages/* 2>/dev/null \
    | sort -rh | head -15 \
    | while read -r size path; do
        printf "  %8s  %s\n" "$size" "$(basename "$path")"
    done

echo ""
echo -e "  ${DIM}These are compiled runtime libraries. Removing them breaks the app.${RESET}"

# ── cleanup ────────────────────────────────────────────────

rm -f /tmp/d2.txt /tmp/df.txt /tmp/d2_paths.txt /tmp/df_paths.txt \
      /tmp/deleted_paths.txt /tmp/shrunk.txt \
      /tmp/pool_del.txt /tmp/pool_shr.txt \
      /tmp/pool_del_tmp.txt /tmp/pool_shr_tmp.txt

echo ""
echo -e "${BOLD}${CYAN}  Done.${RESET}"
echo ""