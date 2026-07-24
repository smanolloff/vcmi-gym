#!/bin/bash

set -euxo pipefail

# For logging
date

# Deletes non-symlinked files older than X hours
# Usage:
#   cleanup 12 /workspace/vcmi-gym/data/v15

HOURS="${1:?Hours required}"
DIR="${2:?Directory required}"

cd "$DIR"
[[ "$HOURS" =~ ^[0-9]+$ ]] || { echo "Invalid hours: $HOURS"; exit 1; }
MINUTES=$(( HOURS * 60 ))

# Associative array filename => 1
# Contains files which are targets of symlinks
declare -A LINK_TARGETS=()
for link in $(find . -maxdepth 1 -type l -printf '%f\n' | sort); do
    target=$(readlink "$link")
    LINK_TARGETS["$target"]=1
done

DELETED=()
for file in $(find . -maxdepth 1 -type f -mmin "+$MINUTES" -printf '%f\n' | sort); do
    [ "${LINK_TARGETS[$file]-}" = "1" ] && continue || :
    rm -f "$file"
    DELETED+=("$file")
done

cat <<EOF
========================
Deleted:
$(printf '%s\n' "${DELETED[@]}")
EOF
