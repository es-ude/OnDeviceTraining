{ pkgs, lib, config, inputs, ... }:

let
  unstablePkgs = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
  # ASan compiler. macOS 26.4+ broke compiler-rt's ASan startup for LLVM
  # <= 21.1.8 (the nixpkgs default that pkgs.clang provides) — __asan_init
  # livelocks before main(). The fix is on LLVM's release/22.x line, so the
  # ASan build links clang 22 from a separately-pinned input. Only the ASan
  # path uses this; the normal build stays on gcc. See docs/CONVENTIONS.md.
  llvm22Pkgs = import inputs.nixpkgs-llvm22 { system = pkgs.stdenv.system; };
  asanClang = llvm22Pkgs.llvmPackages_22.clang;
in
{
  packages = let
    u = unstablePkgs;
  in
    [
      pkgs.git
      u.llvmPackages_21.clang-tools
      pkgs.gcc
      pkgs.clang
      pkgs.cmake
      pkgs.ninja
      u.gcc-arm-embedded-13
      pkgs.zlib
    ];

  languages.c.enable = true;
  #languages.c.compiler = gcc13;
  languages.python = {
    enable = true;
    package = pkgs.python312;
    venv.enable = true;
    venv.quiet = true;
    uv = {
      enable = true;
      package = unstablePkgs.uv;
      sync.enable = true;
      sync.arguments = [ "--all-groups" ];
    };
  };

  scripts = {
    setup_cmake = {
      exec = ''
        CC=gcc cmake --preset unit_test
      '';
      package = pkgs.bash;
      description = "setup cmake";
    };
    clean_cmake = {
      exec = ''
        cmake --build --target clean --preset unit_test
      '';
      package = pkgs.bash;
      description = "clean cmake";
    };
    build_unit_tests = {
      exec = ''
          cmake --build --preset unit_test
      '';
      package = pkgs.bash;
      description = "build unit-tests";
    };
		run_ai_unit_tests = {
			exec = ''
				ctest --preset unit_test
			'';
			package = pkgs.bash;
			description = "Run all Unity unit tests and print their result";
		};
		run_asan_tests = {
			exec = ''
				set -e
				CC=${asanClang}/bin/clang cmake --preset unit_test_asan
				cmake --build --preset unit_test_asan
				ctest --preset unit_test_asan
			'';
			package = pkgs.bash;
			description = "Run the unit-test suite under AddressSanitizer + UBSan";
		};
		ci = {
			exec = ''
				set -e
				# git grep exits 1 on no matches; suspend abort-on-error for this block only
				set +e
				matches=$(git grep -nP '\b(malloc|calloc|realloc|free)\s*\(' \
					-- 'src/' 'test/' \
					':!src/userApi/' \
					':!*.md' \
					| grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|\*)')
				set -e
				if [ -n "$matches" ]; then
					echo "Allocation-locality violation: malloc/calloc/realloc/free are only allowed in src/userApi/."
					echo "All other code must route through reserveMemory/freeReservedMemory in src/userApi/StorageApi.{c,h}."
					echo
					echo "Offending lines:"
					echo "$matches"
					exit 1
				fi
				# #432: examples must step through optimizerStep(), never the raw vtable
				set +e
				matches=$(git grep -nP '(\.|->)step([^A-Za-z0-9_]|$)' \
					-- 'examples/*.c' 'examples/*.h' \
					| grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|\*|/\*)')
				set -e
				if [ -n "$matches" ]; then
					echo "Optimizer-step-entry violation: examples must step the optimizer through optimizerStep() (src/optimizer/include/Optimizer.h)."
					echo "The raw optimizerFunctions[type].step() call runs the same update but fires no ODT_EVENT_OPTIMIZER_* phase events (#432)."
					echo
					echo "Offending lines:"
					echo "$matches"
					exit 1
				fi
				# tracked files must not cite gitignored maintainer-local paths
				set +e
				matches=$(git grep -nE 'docs/superpower[s]/|\.claud[e]/|\.superpower[s]/' \
					-- ':!.gitignore')
				set -e
				if [ -n "$matches" ]; then
					echo "Private-path violation: tracked files must not reference gitignored maintainer-local directories."
					echo "Point at docs/conventions/ (decision registers), an issue/PR number, or drop the citation."
					echo
					echo "Offending lines:"
					echo "$matches"
					exit 1
				fi
				# remat-row-contract (#4): the run block of the remat-row-contract job in
				# .github/workflows/ci.yml, verbatim but for indentation. Outside a git work
				# tree (a jj secondary workspace) git grep cannot scan: skip loudly here, never
				# pass vacuously (the CI job fails closed instead).
				if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
					echo "remat-row-contract: SKIPPED, not a git work tree (jj secondary workspace?); this is NOT a pass"
				else
					set +e
					REMAT=src/userApi/training_loop/remat
					ARM_HOOK='odtHook(Fire|Set)'
					ARM_LAYER='layerFunctions'
					ARM_LOSS='lossFunctions'
					ARM_RNG='(^|[^A-Za-z0-9_])rng[A-Z][A-Za-z0-9_]*[[:space:]]*\('
					ARM_DATA_WRITE='(->|\.)data[[:space:]]*(([-+*/%&|^]|<<|>>)?=([^=]|$)|\+\+|--)'
					ARM_DATA_PREINC='(\+\+|--)[[:space:]]*[A-Za-z_(*][][A-Za-z0-9_.>()*-]*(->|\.)data([^A-Za-z0-9_]|$)'
					PATTERN="$ARM_HOOK|$ARM_LAYER|$ARM_LOSS|$ARM_RNG|$ARM_DATA_WRITE|$ARM_DATA_PREINC"
					# Fail-closed file set: every .c/.h under remat/ is a row file unless excluded
					# here as shared code; include/ is scanned too. Each exclusion is shared,
					# never a row, because:
					#   RematCheck.c         the validator, run by the driver
					#   RematPlan.c          the static-plan generator
					#   RematPlanPolicy.c/.h the plan policies, private to RematPlan
					#   RematScheduler.c     the shared dispatch and the row vtable
					#   RematWireTable.c     the row SDK: the one ->data writer; derives headers
					#                        through layerFunctions at bind
					#   RematCheckedSize.h   size helpers shared by the table, the plan and rows
					#   RematRows.h          the rows' entry points, included by the dispatch
					remat_git() {
						root=$1
						shift
						git -C "$root" "$@" -- "$REMAT/*.c" "$REMAT/*.h" \
							":!$REMAT/RematCheck.c" \
							":!$REMAT/RematPlan.c" \
							":!$REMAT/RematPlanPolicy.c" \
							":!$REMAT/RematPlanPolicy.h" \
							":!$REMAT/RematScheduler.c" \
							":!$REMAT/RematWireTable.c" \
							":!$REMAT/RematCheckedSize.h" \
							":!$REMAT/RematRows.h"
					}
					# Sets hits to the non-comment matches under the repo root $1; returns 2
					# when git grep itself fails (rc 1 only means no match). A comment line
					# starts with //, a /* comment that runs to the line end, or a * followed
					# by a blank, / or the line end; *hdr = ... and /* x */ code are scanned.
					scan() {
						raw=$(remat_git "$1" grep -nE -e "$PATTERN")
						rc=$?
						if [ "$rc" -gt 1 ]; then
							echo "remat-row-contract: git grep failed (rc=$rc) in $1"
							return 2
						fi
						hits=
						if [ -n "$raw" ]; then
							hits=$(printf '%s\n' "$raw" | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|/\*([^*]|\*+[^*/])*(\*+/?)?[[:space:]]*$|\*([[:space:]]|/|$))')
						fi
						return 0
					}
					if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
						echo "remat-row-contract: not a git work tree, so git grep cannot scan; failing closed."
						exit 1
					fi
					gated=$(remat_git . ls-files)
					if [ $? -ne 0 ]; then
						echo "remat-row-contract: git ls-files failed; failing closed."
						exit 1
					fi
					for row in RematArena.c RematHeap.c; do
						case "$gated" in
							*"$REMAT/$row"*) ;;
							*)
								echo "remat-row-contract: $REMAT/$row is not in the gated set; the scan would be vacuous."
								exit 1
								;;
						esac
					done
					scan . || exit 1
					if [ -n "$hits" ]; then
						echo "Remat-row-contract violation: a scheduler row must not fire a hook, run a layer or a loss, call rng*,"
						echo "or write a header's data pointer outside rematWireBind/rematWireRelease."
						echo "See docs/conventions/remat-scheduler.md."
						echo "A shared, non-row file instead joins the exclusion list in this job's run block, with its reason."
						echo
						echo "Offending lines:"
						echo "$hits"
						exit 1
					fi
					# Self-test in a throwaway repo, never in the checkout: each arm must fire on
					# its plant in a row file, a new tracked row file must be gated, and the
					# controls must stay silent.
					(
						tmp=$(mktemp -d) || exit 1
						trap 'rm -rf "$tmp"' EXIT
						git ls-files -z -- "$REMAT" | while IFS= read -r -d "" f; do
							mkdir -p "$tmp/$(dirname "$f")" && cp "$f" "$tmp/$f" || exit 1
						done || exit 1
						git -C "$tmp" init -q && git -C "$tmp" add -A || exit 1
						heap="$tmp/$REMAT/RematHeap.c"
						cp "$heap" "$tmp/heap.orig" || exit 1
						failed=0
						scan "$tmp" || exit 1
						if [ -n "$hits" ]; then
							echo "remat-row-contract self-test: the unplanted copy is not clean"
							failed=1
						fi
						probe() {
							printf '%s\n' "$2" >> "$heap"
							scan "$tmp" || exit 1
							cp "$tmp/heap.orig" "$heap" || exit 1
							if [ "$1" = fire ] && [ -z "$hits" ]; then
								echo "remat-row-contract self-test: the pattern missed: $2"
								failed=1
							elif [ "$1" = silent ] && [ -n "$hits" ]; then
								echo "remat-row-contract self-test: a control line fired: $2"
								failed=1
							fi
						}
						probe fire 'odtHookFire(0);'
						probe fire 'odtHookSet(0, 0);'
						probe fire 'layerFunctions[0].forward(0);'
						probe fire 'lossFunctions[0].forward(0);'
						probe fire 'rngSetSeed (1u);'
						probe fire '/* bind */ h->data = 0;'
						probe fire 'h->data ='
						probe fire 'h->data += 4;'
						probe fire 't.data = 0;'
						probe fire '++(s->wires[i].hdr->data);'
						probe fire '    *hdr = (tensor_t){.data = NULL};'
						probe silent '// rngNextFloat();'
						probe silent ' * h->data = 0;'
						probe silent 'for (i = 0; i < n; ++i) ok = h->data == 0;'
						probe silent 'h->data[0] = 1;'
						probe silent 'freeReservedMemory(b);'
						probe silent 'rng32_t r;'
						# Two arms on one line, so removing any single arm cannot fake this check.
						printf '%s\n' 'odtHookFire(0); h->data = 0;' > "$tmp/$REMAT/RematPlanted.c"
						git -C "$tmp" add "$REMAT/RematPlanted.c" || exit 1
						scan "$tmp" || exit 1
						if [ -z "$hits" ]; then
							echo "remat-row-contract self-test: a new row file is not gated (the file set is no longer fail-closed)"
							failed=1
						fi
						exit "$failed"
					)
					if [ $? -ne 0 ]; then
						echo "remat-row-contract: the self-test failed; the gate no longer catches what it must."
						exit 1
					fi
					echo "remat-row-contract: clean; self-test passed (11 plants and a new row file fire, 6 controls stay silent)."
					set -e
				fi
				# end remat-row-contract
				find src test examples \( -name '*.c' -o -name '*.h' \) -print0 \
					| xargs -0 clang-format --dry-run -Werror
				CC=gcc cmake --preset unit_test
				cmake --build --preset unit_test
				ctest --preset unit_test
				CC=${asanClang}/bin/clang cmake --preset unit_test_asan
				cmake --build --preset unit_test_asan
				ctest --preset unit_test_asan
				ODT_SANITIZER_CC=${asanClang}/bin/clang uv run pytest
			'';
			package = pkgs.bash;
			description = "Run the full CI pipeline locally (format-check + C + ASan/UBSan + Python tests)";
		};

};


  tasks = {
  };

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
      ln -sf "$DEVENV_STATE/venv" "$DEVENV_ROOT/.venv"
    fi
    echo
    echo "Welcome back"
    echo
  '';
}
