from __future__ import annotations

import argparse
import json

from .config import settings
from .rag_pipeline import VisualRAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual-only video RAG pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    index_parser = sub.add_parser("index", help="Index a video")
    index_parser.add_argument("video", type=str, help="Path to input video")

    ask_parser = sub.add_parser("ask", help="Ask question after indexing")
    ask_parser.add_argument("video", type=str, help="Path to input video")
    ask_parser.add_argument("question", type=str, help="User question")

    return parser


def run() -> None:
    args = build_parser().parse_args()
    pipeline = VisualRAGPipeline(settings)

    if args.cmd == "index":
        artifacts = pipeline.index_video(args.video)
        print(json.dumps({"frames": len(artifacts.frames), "chunks": len(artifacts.chunks)}, indent=2))
    elif args.cmd == "ask":
        pipeline.index_video(args.video)
        result = pipeline.answer_question(args.question)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
