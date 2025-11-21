#!/bin/bash
# Demo script showing how to use the visualization tool

echo "====================================="
echo "Visualization Script Demo"
echo "====================================="
echo ""

echo "1. Basic usage - Auto-detect best model:"
echo "   python visualize.py"
echo ""

echo "2. Specify a model explicitly:"
echo "   python visualize.py --model ppo_snake_agent_final.pth"
echo ""

echo "3. Fast playback for quick evaluation:"
echo "   python visualize.py --fps 30 --episodes 3"
echo ""

echo "4. Slow playback for detailed analysis:"
echo "   python visualize.py --fps 5"
echo ""

echo "5. Larger grid with more episodes:"
echo "   python visualize.py --grid_size 15 --episodes 10 --fps 15"
echo ""

echo "6. Get help:"
echo "   python visualize.py --help"
echo ""

echo "====================================="
echo "Note: Make sure you have trained a model first using:"
echo "      python train.py"
echo "====================================="
