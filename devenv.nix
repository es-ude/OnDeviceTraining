{ pkgs, lib, config, inputs, ... }:

let
  unstablePkgs = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  packages = let
    u = unstablePkgs;
  in
    [
      pkgs.git
      pkgs.clang-tools
      pkgs.gcc
      pkgs.cmake
      pkgs.ninja
      u.gcc-arm-embedded-13
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
        cmake --preset unit_test
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
				cmake --preset unit_test_asan
				cmake --build --preset unit_test_asan
				ctest --preset unit_test_asan
			'';
			package = pkgs.bash;
			description = "Run the unit-test suite under AddressSanitizer + UBSan";
		};
		ci = {
			exec = ''
				set -e
				find src test example \( -name '*.c' -o -name '*.h' \) -print0 \
					| xargs -0 clang-format --dry-run -Werror
				cmake --preset unit_test
				cmake --build --preset unit_test
				ctest --preset unit_test
				cmake --preset unit_test_asan
				cmake --build --preset unit_test_asan
				ctest --preset unit_test_asan
				uv run pytest
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
