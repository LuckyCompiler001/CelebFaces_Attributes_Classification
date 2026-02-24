from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent / "assets" / "inputs" / "celeba"
FILES = [
    ROOT / "list_eval_partition.txt",
    ROOT / "list_attr_celeba.txt",
    ROOT / "list_bbox_celeba.txt",
    ROOT / "list_landmarks_align_celeba.txt",
    ROOT / "identity_CelebA.txt",
]


def convert_file(path: Path) -> None:
    """Replace commas with single spaces on every line of the file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    converted = []
    for line in lines:
        if "," in line:
            pieces = [piece.strip() for piece in line.split(",") if piece.strip()]
            converted.append(" ".join(pieces))
        else:
            converted.append(line.rstrip())

    path.write_text("\n".join(converted) + "\n", encoding="utf-8")


def main() -> None:
    for file_path in FILES:
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        convert_file(file_path)


if __name__ == "__main__":
    main()
