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
  languages.python = {
    enable = true;
    package = pkgs.python312;
    venv.enable = true;
    venv.quiet = true;
    uv = {
      enable = true;
      package = unstablePkgs.uv;
      sync.enable = true;
      sync.allExtras = true;
    };
  };

  scripts = {
    setup_cmake = {
      exec = ''
        cmake --preset unit_test
        cmake --preset env5_rev2_debug
        cmake --preset env5_rev2_release
      '';
      package = pkgs.bash;
      description = "setup cmake";
    };
    clean_cmake = {
      exec = ''
        cmake --build --target clean --preset unit_test
        cmake --build --target clean --preset env5_rev2_debug
        cmake --build --target clean --preset env5_rev2_release
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
				build_unit_tests
				TEST_DIR="build/unit_test/test/unit"

				test_results=""

				tests=$(find "$TEST_DIR" -maxdepth 2 -mindepth 2 -type f -executable)

				for test in $tests; do
					test_name=$(basename "$test")
					echo "Running $test_name"

					output=$("$test" 2>&1)
					exitcode=$?

					if [[ $exitcode -eq 0 ]]; then
						test_results+="  \e[32mOK\e[0m    | $test_name"$'\n'
					else
						test_results+="  \e[31mFAIL\e[0m  | $test_name"$'\n'
					fi
				done

				echo "----------------"
				echo -e "Result:\n$test_results"
			'';
			package = pkgs.bash;
			description = "Run all Unity unit tests and print their result";
		};

};


  tasks = {
  };

  enterShell = ''
    echo
    echo "Welcome back"
    echo
  '';
}
