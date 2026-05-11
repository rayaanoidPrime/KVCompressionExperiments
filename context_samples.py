PROSE_CONTEXT = """The global water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth. The total mass of water on Earth remains fairly constant over time, but the partitioning of water into the major reservoirs of ice, fresh water, saline water, and atmospheric water is variable and depends on a wide range of climatic variables. The ocean holds approximately 97 percent of the Earth's water, while the remaining 3 percent is fresh water found in glaciers, ice caps, groundwater, lakes, rivers, and the atmosphere. Of this freshwater, about 68.7 percent is locked in ice caps and glaciers. Water moves from one reservoir to another through physical processes including evaporation, condensation, precipitation, infiltration, surface runoff, and subsurface flow. Evaporation from the ocean surface transfers approximately 434 thousand cubic kilometers of water into the atmosphere each year, and over land, evapotranspiration contributes roughly 71 thousand cubic kilometers. The total precipitation over continents amounts to about 111 thousand cubic kilometers annually, with the difference between precipitation and evapotranspiration driving river discharge back into the oceans. Groundwater represents about 30.1 percent of the world's freshwater, forming one of the largest and most critical reservoirs of liquid fresh water. The average residence time of a water molecule in the atmosphere is approximately 9 days, while residence time in groundwater can range from months to over ten thousand years depending on the depth and lithology of the aquifer. Climate change is significantly altering the water cycle by increasing evaporation rates, changing precipitation patterns, and accelerating glacial melt. A global temperature rise of 1 degree Celsius increases the atmosphere's water-holding capacity by roughly 7 percent according to the Clausius-Clapeyron relation. This intensification of the hydrological cycle leads to more frequent and severe droughts in some regions and intense flooding in others. The Amazon rainforest alone generates approximately 20 billion metric tons of water vapor each day through transpiration, acting as a major regional and global climate regulator. Human activities withdraw approximately 4,000 cubic kilometers of fresh water annually, with agriculture accounting for about 70 percent of all withdrawals. Industrial water usage accounts for about 19 percent, and municipal use accounts for approximately 11 percent of total withdrawals worldwide. A typical shower uses roughly 65 liters of water, while a single irrigation cycle for one hectare of corn requires approximately 5 million liters. Desalination plants currently produce about 95 million cubic meters of fresh water each day globally, though the energy cost remains high at approximately 3.5 kilowatt-hours per cubic meter. Understanding these interconnected processes is fundamental to managing water resources sustainably in a changing climate, particularly as the global population is projected to reach 9.7 billion by the year 2050. Effective water management strategies must account for the full complexity of the hydrologic cycle across local, regional, and global scales to ensure equitable access to clean water for all people."""

CODE_CONTEXT = """
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNK_SIZE: int = 8192


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def serialize_record(record: dict[str, Any], indent: int = 2) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    envelope = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": content_hash,
        "payload": record,
    }
    return json.dumps(envelope, indent=indent)


def archive_old_files(
    directory: Path, max_age_days: int = 90, dry_run: bool = True
) -> list[str]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    now = datetime.now(timezone.utc)
    candidates: list[str] = []
    for fpath in directory.rglob("*"):
        if not fpath.is_file():
            continue
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc)
        age_days = (now - mtime).days
        if age_days > max_age_days:
            candidates.append(str(fpath.relative_to(directory)))
    if not dry_run and candidates:
        archive_dir = directory / "_archive"
        archive_dir.mkdir(exist_ok=True)
        for rel in candidates:
            src = directory / rel
            dst = archive_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
    return candidates
"""

NEEDLE_FACTS = [
    (
        "The mantle reaches temperatures of 3700 degrees Celsius.",
        "What temperature does the mantle reach?",
        "3700",
    ),
    (
        "A typical shower uses roughly 65 liters of water.",
        "How many liters does a typical shower use?",
        "65",
    ),
    (
        "The atmosphere's water-holding capacity increases by 7 percent per degree Celsius.",
        "By what percent does the water-holding capacity increase per degree?",
        "7",
    ),
    (
        "The global population is projected to reach 9.7 billion by the year 2050.",
        "What is the projected global population for 2050?",
        "9.7",
    ),
    (
        "Agriculture accounts for 70 percent of all fresh water withdrawals.",
        "What percent of fresh water withdrawals goes to agriculture?",
        "70",
    ),
]
