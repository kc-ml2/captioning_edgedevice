#!/usr/bin/env python3
"""Publish an immutable ODIC model release to S3.

Authentication is provided exclusively by the AWS CLI credential chain. This
program never accepts or stores IAM access keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MANIFEST = "manifest.json"
REQUIRED = ("model.safetensors.index.json", "tokenizer.model")


class PublishError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[publish-model] {message}", file=sys.stderr)


def run(command: list[str], *, dry_run: bool = False) -> None:
    log("$ " + " ".join(command))
    if dry_run:
        return
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise PublishError(f"명령을 찾지 못했습니다: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise PublishError(f"명령 실행 실패 (exit {error.returncode})") from error


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise PublishError(f"안전하지 않은 모델 경로: {value!r}")
    return Path(*path.parts)


def validate_model(source: Path) -> list[Path]:
    if not source.is_dir():
        raise PublishError(f"모델 디렉터리가 아닙니다: {source}")
    for name in REQUIRED:
        if not (source / name).is_file():
            raise PublishError(f"필수 파일이 없습니다: {name}")
    try:
        index = json.loads((source / REQUIRED[0]).read_text())
        shards = sorted(set(index["weight_map"].values()))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise PublishError("safetensors index를 읽을 수 없습니다.") from error
    if not shards or not all(isinstance(name, str) for name in shards):
        raise PublishError("safetensors index의 weight_map이 잘못되었습니다.")
    for name in shards:
        if not (source / safe_relative(name)).is_file():
            raise PublishError(f"index가 참조하는 shard가 없습니다: {name}")
    return sorted(
        (path for path in source.rglob("*") if path.is_file() and path.name != MANIFEST),
        key=lambda path: path.relative_to(source).as_posix(),
    )


def make_manifest(source: Path, version: str) -> dict:
    files = validate_model(source)
    entries = []
    total = 0
    for number, path in enumerate(files, 1):
        relative = path.relative_to(source).as_posix()
        size = path.stat().st_size
        log(f"SHA-256 ({number}/{len(files)}): {relative}")
        entries.append({"path": relative, "size": size, "sha256": digest(path)})
        total += size
    return {
        "schemaVersion": 1,
        "model": "MobileVLM_V2-1.7B-MLX",
        "version": version,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "totalSize": total,
        "files": entries,
    }


def aws_prefix(profile: str | None, region: str | None) -> list[str]:
    command = ["aws"]
    if profile:
        command += ["--profile", profile]
    if region:
        command += ["--region", region]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="ODIC 모델을 immutable S3 release로 게시합니다.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="odic/models")
    parser.add_argument("--version", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--cloudfront-distribution-id")
    parser.add_argument("--manifest-only", action="store_true", help="업로드 없이 manifest만 생성")
    parser.add_argument("--dry-run", action="store_true", help="AWS 명령을 출력하되 실행하지 않음")
    args = parser.parse_args()

    try:
        source = args.source.resolve()
        manifest = make_manifest(source, args.version)
        prefix = args.prefix.strip("/")
        release_uri = f"s3://{args.bucket}/{prefix}/releases/{args.version}"
        latest = {
            "schemaVersion": 1,
            "version": args.version,
            "manifestPath": f"releases/{args.version}/{MANIFEST}",
        }

        with tempfile.TemporaryDirectory(prefix="odic-release-") as temporary:
            temporary_path = Path(temporary)
            manifest_path = temporary_path / MANIFEST
            latest_path = temporary_path / "latest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            latest_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n")

            if args.manifest_only:
                output = source / MANIFEST
                output.write_text(manifest_path.read_text())
                log(f"manifest 생성 완료: {output}")
                return 0

            aws = aws_prefix(args.profile, args.region)
            run(aws + ["sts", "get-caller-identity"], dry_run=args.dry_run)
            run(aws + ["s3", "sync", str(source), release_uri,
                       "--exclude", MANIFEST, "--no-follow-symlinks", "--no-progress",
                       "--cache-control", "public,max-age=31536000,immutable"], dry_run=args.dry_run)
            # Manifest is the release commit marker and is intentionally uploaded last.
            run(aws + ["s3", "cp", str(manifest_path), f"{release_uri}/{MANIFEST}", "--no-progress",
                       "--content-type", "application/json",
                       "--cache-control", "public,max-age=31536000,immutable"], dry_run=args.dry_run)
            # latest.json is the only mutable object.
            run(aws + ["s3", "cp", str(latest_path), f"s3://{args.bucket}/{prefix}/latest.json", "--no-progress",
                       "--content-type", "application/json",
                       "--cache-control", "no-cache,max-age=0"], dry_run=args.dry_run)
            if args.cloudfront_distribution_id:
                run(aws + ["cloudfront", "create-invalidation",
                           "--distribution-id", args.cloudfront_distribution_id,
                           "--paths", f"/{prefix}/latest.json"], dry_run=args.dry_run)

        log(f"게시 완료: version={args.version}, size={manifest['totalSize']} bytes")
        return 0
    except PublishError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: 사용자가 중단했습니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
