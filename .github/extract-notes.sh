#!/usr/bin/env bash
# Usage: .github/extract-notes.sh OWNER/SLUG GITREF

GITREPO=${1}
GITREF=${2}

if [[ "${GITREF}" == refs/tags/* ]]; then
    TAG="${GITREF#refs/tags/}"
    VERSION="${TAG#v}"
else
    VERSION="Unreleased"
fi

# Extract the release notes from the changelog
sed -n "/## \[${VERSION}\]/, /## /{ /##/!p }" CHANGELOG.md > notes.md

# Add a link to the full changelog
URL="https://github.com/${GITREPO}/blob/main/CHANGELOG.md"
echo "See [CHANGELOG.md](${URL}) for more details." >> notes.md

# Remove leading and trailing empty lines
sed -e :a -e '/./,$!d;/^\n*$/{$d;N;};/\n$/ba' -i notes.md
