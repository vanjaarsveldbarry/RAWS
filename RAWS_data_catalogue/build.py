#!/usr/bin/env python3
import datetime
import pathlib

import yaml

MANIFEST = "manifest.yml"
FIELDS = ("src", "dst", "ini_key")
ROOT = pathlib.Path("CATALOGUE") / "GLOBAL_INPUTS"


def walk(node, prefix=()):
    """Yield (path, entry) for every dataset. A mapping that sets any of FIELDS is a
    dataset, anything else is a folder whose key becomes a path segment.

    Any, not `src` specifically: a dataset that forgot a field is still a dataset,
    and recognising it as one is what lets it raise a KeyError instead of being
    mistaken for a folder full of strings.
    """
    for key, value in node.items():
        if any(f in value for f in FIELDS):
            yield "/".join(prefix + (str(key),)), value
        else:
            yield from walk(value, prefix + (str(key),))


def main():
    man = yaml.safe_load(pathlib.Path(MANIFEST).read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "MANIFEST.yml"
    previous = yaml.safe_load(out_path.read_text()) if out_path.exists() else {}

    today = str(datetime.date.today())
    out = {}
    for name, entry in walk(man):
        src = pathlib.Path(entry["src"])
        # the folder comes from the nesting; dst is just the name within it
        rel = str(pathlib.PurePath(name).parent / entry["dst"])
        if not src.exists():
            raise FileNotFoundError(f"{name}: {src}")   # a link to it would dangle

        dst = ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

        size = (src.stat().st_size if src.is_file() else
                sum(p.stat().st_size for p in src.rglob("*") if p.is_file()))
        out[name] = {**entry, "dst": rel, "kind": "dir" if src.is_dir() else "file",
                     "src_size": size, "built": today}

    # entries dropped from manifest.yml, or given a new dst, must not linger in the tree
    for rel in sorted({e["dst"] for e in previous.values()} - {e["dst"] for e in out.values()}):
        stale = ROOT / rel
        if stale.is_symlink():
            stale.unlink()                  # the link, never the data
        folder = stale.parent
        while folder != ROOT and not any(folder.iterdir()):
            folder.rmdir()
            folder = folder.parent

    out_path.write_text(yaml.safe_dump(out))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
