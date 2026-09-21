"""List available provider models and optionally pin config.yaml model ids."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragc.config import CONFIG_PATH
from ragc.llm.keypool import KeyPool


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def _gemini_models_for_key(key: str) -> list[str]:
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai is not installed") from exc

    client = genai.Client(api_key=key)
    names: list[str] = []
    for model in client.models.list():
        name = getattr(model, "name", "") or ""
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if name:
            names.append(name)
    return sorted(set(names))


def _groq_models_for_key(key: str) -> list[str]:
    try:
        from groq import Groq
    except Exception as exc:
        raise RuntimeError("groq is not installed") from exc

    client = Groq(api_key=key)
    models = client.models.list()
    return sorted(str(item.id) for item in getattr(models, "data", []) if getattr(item, "id", None))


def _choose_flash_lite(models: list[str]) -> str | None:
    candidates = [name for name in models if "flash-lite" in name.lower()]
    if not candidates:
        return None

    def version_key(name: str):
        nums = [int(part) for part in re.findall(r"\d+", name)]
        return nums, name

    return sorted(candidates, key=version_key)[-1]


def _pin_config(path: Path, *, main_model: str | None = None, judge_model: str | None = None) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if main_model:
        data.setdefault("main_llm", {})["model"] = main_model
    if judge_model:
        data.setdefault("judge_llm", {})["model"] = judge_model
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Gemini/Groq model availability for configured API keys.")
    parser.add_argument("--provider", choices=["gemini", "groq", "both"], default="both")
    parser.add_argument("--pin", action="store_true", help="Pin the newest Gemini *flash-lite* model found into config.yaml.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_dotenv()
    config_path = Path(args.config)

    chosen_gemini: str | None = None
    if args.provider in {"gemini", "both"}:
        pool = KeyPool.from_env("gemini")
        if not pool.keys:
            print("Gemini: no GEMINI_API_KEY or GEMINI_API_KEY1 configured")
        for idx, key in enumerate(pool.keys, start=1):
            try:
                models = _gemini_models_for_key(key)
            except Exception as exc:
                print(f"Gemini key {idx}: unavailable ({exc})")
                continue
            print(f"Gemini key {idx}: {len(models)} models")
            for name in models:
                if "flash" in name.lower():
                    print(f"  {name}")
            candidate = _choose_flash_lite(models)
            if candidate:
                chosen_gemini = max([chosen_gemini, candidate] if chosen_gemini else [candidate])

    if args.provider in {"groq", "both"}:
        pool = KeyPool.from_env("groq")
        if not pool.keys:
            print("Groq: no GROQ_API_KEY or GROQ_API_KEY1 configured")
        for idx, key in enumerate(pool.keys, start=1):
            try:
                models = _groq_models_for_key(key)
            except Exception as exc:
                print(f"Groq key {idx}: unavailable ({exc})")
                continue
            print(f"Groq key {idx}: {len(models)} models")
            for name in models:
                print(f"  {name}")

    if args.pin and chosen_gemini:
        _pin_config(config_path, main_model=chosen_gemini)
        print(f"Pinned main_llm.model={chosen_gemini} in {config_path}")
    elif args.pin:
        print("No Gemini flash-lite model found to pin")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
