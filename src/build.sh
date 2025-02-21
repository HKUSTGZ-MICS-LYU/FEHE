#!/bin/bash

# Create target directory if it doesn't exist
TARGET_DIR="/run/user/1000/gvfs/smb-share:server=192.168.2.99,share=xilinx/pynq/overlays/Crypto"
echo "Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Copy and rename the .hwh file
echo "Copying and renaming .hwh file..."
cp "/home/meng/Code/FEHE/hardware/ButterflyPU/ButterflyPU.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh" \
    "$TARGET_DIR/Crypto.hwh"
if [ $? -eq 0 ]; then
    echo "Successfully copied .hwh file to $TARGET_DIR/Crypto.hwh"
else
    echo "Error: Failed to copy .hwh file"
    exit 1
fi

# Copy and rename the .bit file
echo "Copying and renaming .bit file..."
cp "/home/meng/Code/FEHE/hardware/ButterflyPU/ButterflyPU.runs/impl_1/design_1_wrapper.bit" \
    "$TARGET_DIR/Crypto.bit"
if [ $? -eq 0 ]; then
    echo "Successfully copied .bit file to $TARGET_DIR/Crypto.bit"
else
    echo "Error: Failed to copy .bit file"
    exit 1
fi

echo "All operations completed successfully!"