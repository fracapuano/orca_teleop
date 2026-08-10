#!/usr/bin/env bash
# Regenerate the checked-in gRPC stubs for ingress/hand_stream.proto.
#
# The include root MUST be src/ so the descriptor embeds the full package
# path "orca_teleop/ingress/hand_stream.proto". Running protoc from the
# proto's own directory produces a different (and wrong) descriptor blob,
# module name and import alias.
#
# Pinned toolchain: grpcio-tools 1.62.3 (libprotoc 25.1), matching the
# "Protobuf Python Version: 4.25.1" banner in the checked-in files and the
# protobuf>=4.25.9,<5 runtime in pyproject.toml. A different grpcio-tools
# will produce a large, spurious diff.
#
#   ./scripts/gen_proto.sh          regenerate in place
#   ./scripts/gen_proto.sh --check  verify the checked-in files are current
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO="orca_teleop/ingress/hand_stream.proto"

generate_into() {
    uv run python -m grpc_tools.protoc \
        -I. --python_out="$1" --grpc_python_out="$1" "$PROTO"
}

cd "$REPO_ROOT/src"

if [[ "${1:-}" == "--check" ]]; then
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    mkdir -p "$TMP/orca_teleop/ingress"
    generate_into "$TMP"
    status=0
    for f in hand_stream_pb2.py hand_stream_pb2_grpc.py; do
        if ! diff -q "$TMP/orca_teleop/ingress/$f" "orca_teleop/ingress/$f" >/dev/null; then
            echo "STALE: src/orca_teleop/ingress/$f — run ./scripts/gen_proto.sh" >&2
            status=1
        fi
    done
    [[ $status -eq 0 ]] && echo "proto stubs are up to date"
    exit $status
fi

generate_into .
echo "regenerated src/orca_teleop/ingress/hand_stream_pb2{,_grpc}.py"
