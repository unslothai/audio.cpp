#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Stamp an audio.cpp source tree (a checkout of one exact upstream commit) so a
# build from the exported tarball, which has no .git, still reports the real
# commit in `audiocpp_server --version`.
#
# audio.cpp's CMakeLists.txt asks `git rev-parse` for the commit. Outside a
# checkout that answers with whatever repository encloses the tree (on CI: the
# tooling checkout), so the probe is limited to a tree that has its own .git and
# the fallback strings become the resolved commit and its date.
#
# usage: stamp_source.sh <src_dir> <commit-40-hex> <commit-date YYYY-MM-DD>
set -euo pipefail

SRC="${1:?source dir}"
COMMIT="${2:?commit}"
DATE="${3:?commit date}"
printf '%s' "$COMMIT" | grep -qE '^[0-9a-f]{40}$' || { echo "refusing commit '$COMMIT'" >&2; exit 1; }
printf '%s' "$DATE" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' || { echo "refusing date '$DATE'" >&2; exit 1; }
SHORT="${COMMIT:0:7}"
CM="$SRC/CMakeLists.txt"
[ -f "$CM" ] || { echo "no CMakeLists.txt in $SRC" >&2; exit 1; }

# shellcheck disable=SC2016  # ${CMAKE_CURRENT_SOURCE_DIR} is CMake's, not the shell's
sed -i.bak \
  -e "s|^set(AUDIOCPP_GIT_SHA \"unknown\")\$|set(AUDIOCPP_GIT_SHA \"${SHORT}\")|" \
  -e "s|^set(AUDIOCPP_GIT_DATE \"unknown\")\$|set(AUDIOCPP_GIT_DATE \"${DATE}\")|" \
  -e 's|^if (GIT_FOUND)$|if (GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")|' \
  "$CM"
rm -f "$CM.bak"

# shellcheck disable=SC2016
if ! grep -qxF "set(AUDIOCPP_GIT_SHA \"${SHORT}\")" "$CM" \
  || ! grep -qxF "set(AUDIOCPP_GIT_DATE \"${DATE}\")" "$CM" \
  || ! grep -qxF 'if (GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")' "$CM"; then
  echo "CMakeLists.txt lost the git build-info lines this stamp expects" >&2; exit 1
fi

printf '%s\n' "$COMMIT" > "$SRC/UNSLOTH_SOURCE_COMMIT"

# The build source is the immutable UPSTREAM tag, not this tooling checkout.
# Carry only our Windows compatibility block into that ONE source export,
# preserving the rest of upstream's eSpeak patch for every platform. The GPL
# source note includes the actual patch next to the finished server.
ESPEAK_PATCH="$SRC/tools/cmake/espeak_patch.cmake"
if [ -f "$ESPEAK_PATCH" ]; then
  TOOLING_PATCH="$(dirname "${BASH_SOURCE[0]}")/../../tools/cmake/espeak_patch.cmake"
  grep -q '^# BEGIN UNSLOTH_WINDOWS_COMPAT$' "$TOOLING_PATCH"
  grep -q '^# END UNSLOTH_WINDOWS_COMPAT$' "$TOOLING_PATCH"
  WINDOWS_COMPAT="$(sed -n '/^# BEGIN UNSLOTH_WINDOWS_COMPAT$/,/^# END UNSLOTH_WINDOWS_COMPAT$/p' "$TOOLING_PATCH")"
  sed -i.bak '/^# BEGIN UNSLOTH_WINDOWS_COMPAT$/,/^# END UNSLOTH_WINDOWS_COMPAT$/d' "$ESPEAK_PATCH"
  rm -f "$ESPEAK_PATCH.bak"
  printf '\n' >> "$ESPEAK_PATCH"
  printf '%s\n' "$WINDOWS_COMPAT" >> "$ESPEAK_PATCH"
fi
echo "stamped $SRC: git ${SHORT} ${DATE}"
