# Activate the openmhc venv on Sherlock.
#
# Source this from an interactive shell:
#     source scripts/dev/activate-openmhc.sh
#
# Idempotent; safe to source multiple times per session. Sets the SSL+python
# shared-library path that Sherlock's /share/software python builds need at
# runtime, then activates the venv at $SCRATCH/envs/openmhc.

PYTHON_LIB="/share/software/user/open/python/3.12.1/lib"
OPENSSL_LIB="/share/software/user/open/openssl/3.0.7/lib64"
VENV="${SCRATCH:-$HOME}/envs/openmhc"

# Strip the Sherlock shell's PYTHONPATH — it points at py3.9 numpy/scipy/torch
# packages which silently override the venv's own site-packages and break
# every wheel build (notably pandas). The venv is fully self-contained, so
# PYTHONPATH should be empty inside it.
unset PYTHONPATH

# Strip MPI compiler wrappers (Sherlock default sets CC=mpicc, CXX=mpic++,
# F77=mpif77, FC=mpifort). These confuse pip's source builds (numpy, pandas,
# xgboost, etc. fail) — let pip pick the system gcc instead.
unset CC CXX F77 FC

# Prepend our libs only if not already present, so re-sourcing doesn't grow
# LD_LIBRARY_PATH unboundedly.
case ":${LD_LIBRARY_PATH:-}:" in
  *":$PYTHON_LIB:"*) ;;
  *) export LD_LIBRARY_PATH="$PYTHON_LIB:${LD_LIBRARY_PATH:-}" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
  *":$OPENSSL_LIB:"*) ;;
  *) export LD_LIBRARY_PATH="$OPENSSL_LIB:${LD_LIBRARY_PATH}" ;;
esac

export PIP_CACHE_DIR="${SCRATCH:-$HOME}/.cache/pip"

if [ ! -d "$VENV" ]; then
  echo "openmhc venv not found at $VENV." >&2
  echo "Create it with:" >&2
  echo "  LD_LIBRARY_PATH=\"$PYTHON_LIB:$OPENSSL_LIB\" \\" >&2
  echo "    /share/software/user/open/python/3.12.1/bin/python3 -m venv \"$VENV\"" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "Activated openmhc venv: $(python --version 2>&1) at $VENV"
