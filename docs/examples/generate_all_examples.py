#!/usr/bin/env python3
"""
Generate All Examples

This script runs all the example scripts in the docs/examples directory,
creating a complete collection of drift-datasets demonstrations.
Perfect for testing, documentation generation, or comprehensive evaluation.

Expected output:
- All example datasets generated
- All visualizations created
- All analysis results exported
- Summary report of generated examples
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def setup_output_directory():
    """Create the output directory for all examples."""

    output_dir = Path("docs/examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for organized output
    subdirs = ["expertsystems", "analysis", "visualizations", "datasets"]
    for subdir in subdirs:
        (output_dir / subdir).mkdir(exist_ok=True)

    print(f"📁 Output directory prepared: {output_dir}")
    return output_dir


def get_example_scripts():
    """Get list of all example scripts to run."""

    examples_dir = Path("docs/examples")

    # Define the order of execution (basic examples first)
    ordered_examples = [
        "basic_synthetic.py",
        "basic_real_world.py",
        "gradual_drift.py",
        "expertsystems_reproduction.py",
        "drift_analysis.py",
    ]

    # Find all Python scripts in examples directory
    all_scripts = list(examples_dir.glob("*.py"))
    all_scripts = [script for script in all_scripts if script.name != "generate_all_examples.py"]

    # Start with ordered examples, then add any additional ones
    example_scripts = []
    for script_name in ordered_examples:
        script_path = examples_dir / script_name
        if script_path.exists():
            example_scripts.append(script_path)

    # Add any additional scripts not in the ordered list
    for script in all_scripts:
        if script not in example_scripts:
            example_scripts.append(script)

    return example_scripts


def run_example_script(script_path, timeout=300):
    """Run a single example script with timeout and error handling."""

    print(f"\n🚀 Running {script_path.name}...")
    print("-" * 50)

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent.parent,  # Run from project root
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        end_time = time.time()
        execution_time = end_time - start_time

        if result.returncode == 0:
            print(f"✅ {script_path.name} completed successfully in {execution_time:.1f}s")
            return {
                "script": script_path.name,
                "status": "success",
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        else:
            print(f"❌ {script_path.name} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return {
                "script": script_path.name,
                "status": "failed",
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

    except subprocess.TimeoutExpired:
        print(f"⏰ {script_path.name} timed out after {timeout}s")
        return {
            "script": script_path.name,
            "status": "timeout",
            "execution_time": timeout,
            "error": f"Script timed out after {timeout} seconds",
        }

    except Exception as e:
        print(f"💥 {script_path.name} crashed with exception: {e}")
        return {"script": script_path.name, "status": "crashed", "execution_time": time.time() - start_time, "error": str(e)}


def generate_summary_report(results, output_dir):
    """Generate a comprehensive summary report of all examples."""

    print(f"\n📊 Generating summary report...")

    # Count results by status
    status_counts = {}
    total_time = 0

    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        total_time += result.get("execution_time", 0)

    # Create summary statistics
    summary = {
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_examples": len(results),
        "total_execution_time": round(total_time, 2),
        "status_summary": status_counts,
        "success_rate": round((status_counts.get("success", 0) / len(results)) * 100, 1),
        "example_results": results,
    }

    # Export summary as JSON
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Create human-readable report
    report_path = output_dir / "generation_report.md"

    with open(report_path, "w") as f:
        f.write("# drift-datasets Examples Generation Report\n\n")
        f.write(f"Generated: {summary['generation_timestamp']}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total Examples**: {summary['total_examples']}\n")
        f.write(f"- **Success Rate**: {summary['success_rate']}%\n")
        f.write(f"- **Total Execution Time**: {summary['total_execution_time']}s\n\n")

        f.write("## Status Breakdown\n\n")
        for status, count in status_counts.items():
            emoji = {"success": "✅", "failed": "❌", "timeout": "⏰", "crashed": "💥"}.get(status, "❓")
            f.write(f"- {emoji} **{status.title()}**: {count} examples\n")

        f.write("\n## Individual Results\n\n")

        for result in results:
            status = result["status"]
            emoji = {"success": "✅", "failed": "❌", "timeout": "⏰", "crashed": "💥"}.get(status, "❓")

            f.write(f"### {emoji} {result['script']}\n\n")
            f.write(f"- **Status**: {status.title()}\n")
            f.write(f"- **Execution Time**: {result.get('execution_time', 0):.1f}s\n")

            if status == "failed":
                f.write(f"- **Return Code**: {result.get('return_code', 'N/A')}\n")
                if result.get("stderr"):
                    f.write(f"- **Error**: {result['stderr'][:200]}...\n")

            elif status in ["timeout", "crashed"]:
                f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")

            f.write("\n")

    print(f"   Summary JSON: {summary_path}")
    print(f"   Report markdown: {report_path}")

    return summary


def check_generated_files(output_dir):
    """Check what files were generated by all examples."""

    print(f"\n📋 Checking generated files...")

    file_count = 0
    file_types = {}

    for file_path in output_dir.rglob("*"):
        if file_path.is_file() and file_path.name not in ["generation_summary.json", "generation_report.md"]:
            file_count += 1
            suffix = file_path.suffix.lower()
            file_types[suffix] = file_types.get(suffix, 0) + 1

    print(f"   Total files generated: {file_count}")

    if file_types:
        print("   File types:")
        for file_type, count in sorted(file_types.items()):
            file_type_display = file_type if file_type else "(no extension)"
            print(f"     {file_type_display}: {count} files")

    return file_count, file_types


def main():
    """Main function to generate all examples."""

    print("🚀 drift-datasets Examples Generation Suite")
    print("=" * 50)
    print("This script will run all example scripts and generate")
    print("a comprehensive collection of datasets, visualizations,")
    print("and analysis results.\n")

    # Check if we're in the right directory
    if not Path("docs/examples").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   (the directory containing 'docs/examples/')")
        sys.exit(1)

    # Setup
    output_dir = setup_output_directory()
    example_scripts = get_example_scripts()

    if not example_scripts:
        print("❌ No example scripts found in docs/examples/")
        sys.exit(1)

    print(f"📝 Found {len(example_scripts)} example scripts:")
    for script in example_scripts:
        print(f"   • {script.name}")

    # Confirm execution
    print(f"\n⚠️  This will take several minutes to complete.")
    print(f"   Each script has a timeout of 5 minutes.")

    response = input("\nProceed with generation? (y/n): ").lower().strip()
    if response not in ["y", "yes"]:
        print("Generation cancelled.")
        sys.exit(0)

    print("\n" + "=" * 50)
    print("🏃 Starting example generation...")

    # Run all examples
    results = []
    start_total_time = time.time()

    for i, script_path in enumerate(example_scripts, 1):
        print(f"\n[{i}/{len(example_scripts)}]", end=" ")
        result = run_example_script(script_path)
        results.append(result)

        # Brief pause between scripts
        time.sleep(1)

    end_total_time = time.time()
    total_execution_time = end_total_time - start_total_time

    print("\n" + "=" * 50)
    print("📊 Generation Complete!")

    # Generate summary report
    summary = generate_summary_report(results, output_dir)

    # Check generated files
    file_count, file_types = check_generated_files(output_dir)

    # Print final summary
    print(f"\n✅ Example Generation Summary:")
    print(f"   • Examples run: {len(results)}")
    print(f"   • Successful: {summary['status_summary'].get('success', 0)}")
    print(f"   • Failed: {summary['status_summary'].get('failed', 0)}")
    print(f"   • Success rate: {summary['success_rate']}%")
    print(f"   • Total time: {total_execution_time:.1f}s")
    print(f"   • Files generated: {file_count}")

    if summary["success_rate"] == 100:
        print("\n🎉 All examples completed successfully!")
    elif summary["success_rate"] >= 75:
        print(f"\n✅ Most examples completed successfully ({summary['success_rate']}%)")
    else:
        print(f"\n⚠️  Many examples failed ({100 - summary['success_rate']:.1f}% failure rate)")
        print("   Check the detailed report for troubleshooting information.")

    print(f"\n📁 All outputs available in: {output_dir}")
    print("   Check generation_report.md for detailed results.")

    print("\n🚀 Examples ready for use!")
    print("Next steps:")
    print("1. Explore the generated datasets and visualizations")
    print("2. Use the examples as templates for your own projects")
    print("3. Modify configurations to experiment with different scenarios")
    print("4. Integrate the datasets with your drift detection algorithms")


if __name__ == "__main__":
    main()
