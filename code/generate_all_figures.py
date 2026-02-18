"""
Master Script: Generate All Figures for QRC Paper
Runs all figure generation scripts in sequence
"""

import subprocess
import sys
import os

# Change to code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("GENERATING ALL FIGURES FOR QRC PAPER")
print("="*60)

figure_scripts = [
    ('fig2_layer_sweep.py', 'Figure 2: Layer Sweep Results (HERO FIGURE)'),
    ('fig3_stability.py', 'Figure 3: Stability Analysis'),
    ('fig4_walkforward.py', 'Figure 4: Walk-Forward Cross-Validation'),
    ('fig5_horizon.py', 'Figure 5: Prediction Horizon Decay'),
    ('fig6_comparison.py', 'Figure 6: Cross-Platform Comparison'),
]

successful = []
failed = []

for script, description in figure_scripts:
    print(f"\n{'='*60}")
    print(f"Generating: {description}")
    print(f"Script: {script}")
    print('='*60)

    try:
        # Run script
        result = subprocess.run([sys.executable, script],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(result.stdout)
            successful.append((script, description))
            print(f"✓ SUCCESS: {description}")
        else:
            print(f"✗ FAILED: {description}")
            print("STDERR:", result.stderr)
            failed.append((script, description))

    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {description}")
        failed.append((script, description))
    except Exception as e:
        print(f"✗ ERROR: {description}")
        print(f"Exception: {e}")
        failed.append((script, description))

# Summary
print(f"\n{'='*60}")
print("GENERATION SUMMARY")
print('='*60)
print(f"\nSuccessful: {len(successful)}/{len(figure_scripts)}")
for script, desc in successful:
    print(f"  ✓ {desc}")

if failed:
    print(f"\nFailed: {len(failed)}/{len(figure_scripts)}")
    for script, desc in failed:
        print(f"  ✗ {desc}")
else:
    print("\n🎉 ALL FIGURES GENERATED SUCCESSFULLY!")

print(f"\n{'='*60}")
print("Figures saved to: ../figures/")
print("Formats: PDF (vector) and PNG (raster, 300 DPI)")
print('='*60)
