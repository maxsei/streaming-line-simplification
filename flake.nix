{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/release-24.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        fhs = let my-python = pkgs.python311;
        in pkgs.buildFHSUserEnv {
          name = "fhs-shell";
          targetPkgs = p: with p; [ zlib libcxx my-python poetry ];
        };
      in {
        devShell = fhs.env;
      });
}
