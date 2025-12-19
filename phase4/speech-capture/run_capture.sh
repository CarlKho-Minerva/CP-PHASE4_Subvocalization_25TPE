#!/bin/bash
# ============================================================================
# SPEECH SPECTRUM CAPTURE TOOL
# Run mouthing (training data) OR subvocalization (testing data)
# ============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          SPEECH SPECTRUM DATA CAPTURE                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  1) Run MOUTHING capture (Level 3 - Training Data)           ║"
echo "║  2) Run SUBVOCAL capture (Level 4 - Testing Data)            ║"
echo "║  3) Compile both tools from source                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
read -p "Select option [1/2/3]: " choice

case $choice in
    1)
        if [ ! -f "./capture_mouthing" ]; then
            echo "Compiling capture_mouthing.cpp..."
            g++ -std=c++17 -O2 capture_mouthing.cpp -o capture_mouthing -framework IOKit -framework CoreFoundation
        fi
        ./capture_mouthing
        ;;
    2)
        if [ ! -f "./capture_subvocal" ]; then
            echo "Compiling capture_subvocal.cpp..."
            g++ -std=c++17 -O2 capture_subvocal.cpp -o capture_subvocal -framework IOKit -framework CoreFoundation
        fi
        ./capture_subvocal
        ;;
    3)
        echo "Compiling both tools..."
        g++ -std=c++17 -O2 capture_mouthing.cpp -o capture_mouthing -framework IOKit -framework CoreFoundation
        g++ -std=c++17 -O2 capture_subvocal.cpp -o capture_subvocal -framework IOKit -framework CoreFoundation
        echo "Done! Run this script again and select 1 or 2."
        ;;
    *)
        echo "Invalid option."
        ;;
esac
