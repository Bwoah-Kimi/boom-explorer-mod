"""
Generate requirements.txt with only the packages actually used in this project
"""

import subprocess
import sys

# Core packages used in this project
CORE_PACKAGES = [
    'numpy',
    'scipy', 
    'pandas',
    'torch',
    'gpytorch',
    'botorch',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'openpyxl',
    'xlrd',
    'PyYAML',
    'tqdm',
    'colorama',
]

def get_installed_version(package):
    """Get the installed version of a package"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        return None

def main():
    print("Generating requirements.txt with actual installed versions...\n")
    
    requirements = []
    missing = []
    
    for package in CORE_PACKAGES:
        version = get_installed_version(package)
        if version:
            requirements.append(f"{package}=={version}")
            print(f"✓ {package}=={version}")
        else:
            missing.append(package)
            print(f"✗ {package} - NOT INSTALLED")
    
    # Write requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write("# BOOM Explorer Requirements\n")
        f.write("# Generated automatically - do not edit manually\n\n")
        f.write("# Core scientific computing\n")
        for req in requirements[:3]:
            f.write(f"{req}\n")
        f.write("\n# Machine Learning / Optimization\n")
        for req in requirements[3:7]:
            f.write(f"{req}\n")
        f.write("\n# Visualization\n")
        for req in requirements[7:9]:
            f.write(f"{req}\n")
        f.write("\n# File handling\n")
        for req in requirements[9:11]:
            f.write(f"{req}\n")
        f.write("\n# Configuration and utilities\n")
        for req in requirements[11:]:
            f.write(f"{req}\n")
    
    print(f"\n✓ Generated requirements.txt with {len(requirements)} packages")
    
    if missing:
        print(f"\n⚠️  Warning: {len(missing)} packages not installed:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nRun: pip install " + " ".join(missing))

if __name__ == "__main__":
    main()