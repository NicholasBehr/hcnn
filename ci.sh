#!/bin/bash
# local ci-test script
set -e

uv run ruff check
uv run pyright