#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-${(%):-%x}}")" &> /dev/null && pwd)
sudo mount -t tmpfs -o size=4G,noatime tmpfs $SCRIPT_DIR/target
# If cargo hits disk space limit, clear the cache. We should reduce the build size instead of increasing the tmpfs size.
