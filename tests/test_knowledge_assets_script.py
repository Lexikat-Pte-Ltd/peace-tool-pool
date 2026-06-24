import subprocess
from pathlib import Path


def test_install_knowledge_assets_script_copies_expected_files(tmp_path):
    source = tmp_path / "peace"
    source_knowledge = source / "dependencies" / "knowledge"
    source_knowledge.mkdir(parents=True)
    for name in (
        "k2_rock_type.json",
        "k2_rock_age.json",
        "earthquake_1970_4.5mag.csv",
        "gem_active_faults_harmonized.geojson",
    ):
        (source_knowledge / name).write_text("fixture\n", encoding="utf-8")

    dest = tmp_path / "dest-knowledge"
    result = subprocess.run(
        [
            "bash",
            "scripts/install_knowledge_assets.sh",
            "--source",
            str(source),
            "--dest",
            str(dest),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (dest / "k2_rock_type.json").read_text(encoding="utf-8") == "fixture\n"
    assert (dest / "gem_active_faults_harmonized.geojson").exists()
