#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-${(%):-%x}}")" &> /dev/null && pwd)
sudo mount -t tmpfs -o size=12G,noatime tmpfs $SCRIPT_DIR/target

# To run passwordless, add below to /etc/sudoers (use visudo): 
# username ALL=(ALL) NOPASSWD: /path/to/docent/mount_tmpfs.sh
