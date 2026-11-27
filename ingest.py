"""CLI for the spec library.

    python ingest.py data/axi4_spec.pdf --name "AXI4"
    python ingest.py --list
    python ingest.py --delete "AXI4"
"""
import argparse
import os
import sys

from specs import delete_spec, ingest_pdf, list_specs, resolve_spec


def main():
    parser = argparse.ArgumentParser(description="Ingest a specification PDF into the spec library.")
    parser.add_argument("pdf", nargs="?", help="Path to the spec PDF")
    parser.add_argument("--name", help="Display name for the spec (default: PDF file name)")
    parser.add_argument("--skip-pages", type=int, default=0,
                        help="Force-skip the first N pages (TOC pages are already auto-skipped)")
    parser.add_argument("--list", action="store_true", help="List ingested specs")
    parser.add_argument("--delete", metavar="NAME", help="Remove a spec from the library")
    args = parser.parse_args()

    if args.list:
        specs = list_specs()
        if not specs:
            print("No specs ingested yet.")
        for s in specs:
            print(f"{s['name']:<30} {s['pages']:>5} pages  {s['chunks']:>6} chunks  ({s['source_file']})")
        return

    if args.delete:
        spec = resolve_spec(args.delete)
        delete_spec(spec["id"])
        print(f"✓ Removed '{spec['name']}'")
        return

    if not args.pdf:
        parser.error("provide a PDF path, or use --list / --delete")
    if not os.path.isfile(args.pdf):
        sys.exit(f"File not found: {args.pdf}")

    name = args.name or os.path.splitext(os.path.basename(args.pdf))[0].replace("_", " ")
    last = {"msg": None}

    def progress(frac, msg):
        if msg != last["msg"]:
            print(f"  [{frac:>4.0%}] {msg}", end="\r" if msg.startswith("Embedding") else "\n")
            last["msg"] = msg

    print(f"Ingesting {args.pdf} as '{name}'...")
    spec = ingest_pdf(args.pdf, name, skip_first_pages=args.skip_pages, progress=progress)
    print(f"\n✓ '{spec['name']}': indexed {spec['pages_indexed']}/{spec['pages']} pages "
          f"into {spec['chunks']} chunks")


if __name__ == "__main__":
    main()
